import pandas as pd
import numpy as np
import os
import warnings
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import traceback
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Config:
    IMAGE_DIR = 'fashion-dataset/images'
    STYLES_FILE = 'styles.csv'
    SHAPE_CACHE_FILE = 'shape_features_cache.pkl'

class ShapeFeatureExtractor:
    """Extract shape features dari gambar fashion items"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def extract_shape_features(self, image_path):
        """Extract comprehensive shape features"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Cannot load image: {image_path}")
                return None
            
            # Resize untuk konsistensi
            img = cv2.resize(img, (200, 200))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection untuk mendapatkan bentuk
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Jika tidak ada contour, gunakan seluruh image
                contours, _ = cv2.findContours(np.ones_like(gray) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Extract comprehensive shape features
            features = {}
            
            # 1. Basic geometric features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            features['aspect_ratio'] = w / h if h > 0 else 0
            features['area_ratio'] = area / (w * h) if w * h > 0 else 0
            features['solidity'] = area / cv2.contourArea(cv2.convexHull(largest_contour)) if cv2.contourArea(cv2.convexHull(largest_contour)) > 0 else 0
            features['extent'] = area / (w * h) if w * h > 0 else 0
            
            # 2. Hu Moments (shape descriptors)
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Normalize Hu moments
            for i in range(7):
                if hu_moments[i] != 0:
                    features[f'hu_{i}'] = -np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
                else:
                    features[f'hu_{i}'] = 0
            
            # 3. Additional shape features
            features['compactness'] = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 4. Bounding box ratios
            features['width_ratio'] = w / 200.0  # normalized by image width
            features['height_ratio'] = h / 200.0  # normalized by image height
            
            # 5. Position in image
            features['center_x'] = (x + w/2) / 200.0
            features['center_y'] = (y + h/2) / 200.0
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting shape features: {e}")
            return None

class ProductCategorizer:
    """Klasifikasi kategori produk berdasarkan nama"""
    
    def __init__(self):
        # Keywords untuk setiap kategori
        self.category_keywords = {
            'shoes': ['shoe', 'sneaker', 'boot', 'footwear', 'loafer', 'pump', 'heel', 'oxford'],
            'sandals': ['sandal', 'flipflop', 'slide', 'thong', 'flip-flop'],
            'dresses': ['dress', 'gown', 'frock', 'jumper'],
            'shirts': ['shirt', 'top', 'blouse', 't-shirt', 'tshirt', 'polo'],
            'pants': ['pant', 'trouser', 'jeans', 'legging', 'chino'],
            'bags': ['bag', 'purse', 'backpack', 'handbag', 'clutch', 'tote'],
            'accessories': ['hat', 'cap', 'belt', 'scarf', 'tie', 'watch', 'gloves'],
            'jackets': ['jacket', 'coat', 'blazer', 'hoodie', 'sweater'],
            'skirts': ['skirt', 'mini', 'midi', 'maxi'],
            'shorts': ['short', 'bermuda', 'trunks']
        }
    
    def predict_category(self, product_name=""):
        """Predict kategori dari nama produk"""
        if not product_name:
            return 'unknown'
        
        product_name_lower = product_name.lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in product_name_lower:
                    return category
        
        return 'unknown'

class UltraAccurateShapeMatcher:
    """Shape matcher yang akurat berdasarkan bentuk produk"""
    
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.feature_extractor = ShapeFeatureExtractor()
        self.categorizer = ProductCategorizer()
        self.features_cache = {}
        self.image_mapping = {}
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize system dengan precomputed features"""
        print("🔄 Initializing ULTRA ACCURATE Shape-Based System...")
        
        # Load cache jika ada
        if os.path.exists(self.config.SHAPE_CACHE_FILE):
            print("📦 Loading precomputed shape features...")
            try:
                with open(self.config.SHAPE_CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.features_cache = cache_data.get('features_cache', {})
                    self.image_mapping = cache_data.get('image_mapping', {})
                print(f"✅ Loaded {len(self.features_cache)} precomputed shape features")
            except Exception as e:
                print(f"❌ Cache corrupted, rebuilding... {e}")
                self._build_features_cache()
        else:
            self._build_features_cache()
    
    def _build_features_cache(self):
        """Build cache untuk semua gambar"""
        print("🔄 Building shape features cache...")
        
        self._build_image_mapping()
        
        valid_count = 0
        for product_id, filename in tqdm(self.image_mapping.items(), desc="Extracting shapes"):
            image_path = os.path.join(self.config.IMAGE_DIR, filename)
            
            if os.path.exists(image_path):
                # Extract shape features
                shape_features = self.feature_extractor.extract_shape_features(image_path)
                
                if shape_features:
                    # Get product info untuk kategori
                    product_row = self.df[self.df['id'].astype(str) == product_id]
                    product_name = ""
                    if not product_row.empty:
                        product_name = product_row.iloc[0].get('productDisplayName', '')
                    
                    # Predict kategori
                    category = self.categorizer.predict_category(product_name)
                    
                    self.features_cache[product_id] = {
                        'shape_features': shape_features,
                        'category': category,
                        'product_name': product_name
                    }
                    valid_count += 1
        
        # Save cache
        cache_data = {
            'features_cache': self.features_cache,
            'image_mapping': self.image_mapping
        }
        with open(self.config.SHAPE_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ Built cache with {valid_count} shape features")
        
        # Print category distribution
        self._print_category_stats()
    
    def _print_category_stats(self):
        """Print statistics kategori"""
        categories = {}
        for data in self.features_cache.values():
            cat = data['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("📊 Category Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count} products")
    
    def _build_image_mapping(self):
        """Build mapping product ID ke image files"""
        if not os.path.exists(self.config.IMAGE_DIR):
            print("⚠️ Images directory not found")
            return
        
        try:
            image_files = [f for f in os.listdir(self.config.IMAGE_DIR) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            print(f"📁 Found {len(image_files)} image files")
            
            for file in image_files:
                name_without_ext = os.path.splitext(file)[0]
                self.image_mapping[name_without_ext] = file
            
            print(f"✅ Mapped {len(self.image_mapping)} products to images")
        except Exception as e:
            print(f"❌ Error building image mapping: {e}")
    
    def _calculate_shape_similarity(self, features1, features2):
        """Hitung similarity antara dua shape features"""
        if not features1 or not features2:
            return 0.0
        
        try:
            # Get common features
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 0.0
            
            # Convert to vectors
            vec1 = np.array([features1[k] for k in common_keys])
            vec2 = np.array([features2[k] for k in common_keys])
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            
            # Ensure valid range
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0
    
    def recommend_by_image(self, image_path, n_recommendations=None):
        """Rekomendasi berdasarkan bentuk gambar - ULTRA AKURAT"""
        try:
            print("🎯 ULTRA ACCURATE SHAPE-BASED MATCHING...")
            print(f"📁 Processing image: {image_path}")
            
            # Step 1: Extract features dari gambar upload
            uploaded_features = self.feature_extractor.extract_shape_features(image_path)
            if not uploaded_features:
                print("❌ Cannot extract shape features from uploaded image")
                return {"error": "Cannot extract shape features from uploaded image"}
            
            print(f"✅ Extracted {len(uploaded_features)} shape features")
            
            # Step 2: Cari produk dengan shape yang mirip
            similar_products = []
            processed_count = 0
            
            for product_id, cache_data in self.features_cache.items():
                product_features = cache_data['shape_features']
                product_category = cache_data['category']
                
                # Hitung similarity shape
                similarity = self._calculate_shape_similarity(uploaded_features, product_features)
                
                if similarity > 0.3:  # Threshold minimum
                    similar_products.append({
                        'product_id': product_id,
                        'similarity': similarity,
                        'category': product_category
                    })
                    processed_count += 1
            
            print(f"🔍 Found {len(similar_products)} similar products (processed {processed_count})")
            
            # Step 3: Urutkan berdasarkan similarity tertinggi
            similar_products.sort(key=lambda x: x['similarity'], reverse=True)
            
            if n_recommendations is None:
                n_recommendations = min(100, len(similar_products))
            
            # Step 4: Siapkan hasil rekomendasi
            recommendations = []
            for i, item in enumerate(similar_products[:n_recommendations]):
                product_id = item['product_id']
                similarity_score = float(item['similarity'] * 100)
                
                # Cari data produk
                product_row = self.df[self.df['id'].astype(str) == product_id]
                if not product_row.empty:
                    product = product_row.iloc[0]
                    
                    recommendations.append({
                        'id': int(product['id']) if pd.notna(product['id']) else 0,
                        'productDisplayName': str(product['productDisplayName']) if pd.notna(product['productDisplayName']) else 'Unknown',
                        'articleType': str(product['articleType']) if pd.notna(product['articleType']) else 'Unknown',
                        'baseColour': str(product['baseColour']) if pd.notna(product['baseColour']) else 'Unknown',
                        'season': str(product.get('season', '')) if pd.notna(product.get('season', '')) else '',
                        'usage': str(product.get('usage', '')) if pd.notna(product.get('usage', '')) else '',
                        'similarity_score': round(similarity_score, 2),
                        'exact_match': bool(similarity_score > 70),
                        'has_image': True,
                        'rank': i + 1,
                        'match_type': 'shape_similarity',
                        'product_category': item['category']
                    })
            
            print(f"✅ Shape matching completed! Found {len(recommendations)} similar products")
            
            # Debug info DETAILED
            if recommendations:
                print(f"🎯 TOP 10 SHAPE MATCHES:")
                for i in range(min(10, len(recommendations))):
                    rec = recommendations[i]
                    print(f"   {i+1}. {rec['productDisplayName'][:50]}...")
                    print(f"       Score: {rec['similarity_score']}% | Type: {rec['articleType']} | Category: {rec['product_category']}")
            else:
                print("❌ NO SHAPE-BASED MATCHES FOUND!")
            
            return recommendations if recommendations else {"error": "No similar products found"}
            
        except Exception as e:
            error_msg = f"Error in shape matching: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return {"error": error_msg}

class SmartTextRecommender:
    """Text recommender yang smart dengan understanding kategori"""
    
    def __init__(self, df):
        self.df = df
        self.categorizer = ProductCategorizer()
        self._build_text_similarity_matrix()
    
    def _build_text_similarity_matrix(self):
        """Build text similarity matrix"""
        print("🔄 Building smart text similarity matrix...")
        
        try:
            # Enhanced feature combination
            self.df['search_features'] = (
                self.df['productDisplayName'].fillna('') + ' ' +
                self.df['articleType'].fillna('') + ' ' +
                self.df['baseColour'].fillna('') + ' ' +
                self.df.get('season', '').fillna('') + ' ' +
                self.df.get('usage', '').fillna('')
            )
            
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=8000,
                min_df=2
            )
            
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['search_features'])
            print(f"✅ Text similarity matrix built: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            print(f"❌ Error building text similarity: {e}")
            raise
    
    def recommend_by_text(self, query_text, n_recommendations=None):
        """Rekomendasi text dengan understanding kategori"""
        try:
            print(f"🔍 Smart text search: '{query_text}'")
            
            # Predict kategori dari query
            query_category = self.categorizer.predict_category(query_text)
            if query_category != 'unknown':
                print(f"📋 Detected category from query: {query_category}")
            
            # Transform query
            query_vec = self.tfidf.transform([query_text.lower()])
            
            # Calculate similarity
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get relevant results
            relevant_indices = np.where(similarity_scores > 0.01)[0]
            
            if not relevant_indices.size:
                return {"error": f"No products found for '{query_text}'"}
            
            # Sort by similarity
            sorted_indices = relevant_indices[np.argsort(similarity_scores[relevant_indices])[::-1]]
            
            if n_recommendations is None:
                n_recommendations = min(100, len(sorted_indices))
            
            results = self.df.iloc[sorted_indices[:n_recommendations]]
            similarity_values = similarity_scores[sorted_indices[:n_recommendations]]
            
            recommendations = []
            for idx, (_, product) in enumerate(results.iterrows()):
                product_name = str(product['productDisplayName']) if pd.notna(product['productDisplayName']) else ''
                product_category = self.categorizer.predict_category(product_name)
                
                # Base similarity score
                base_score = float(similarity_values[idx] * 100)
                
                # Boost score jika kategori cocok
                if query_category != 'unknown' and product_category == query_category:
                    base_score = min(100, base_score * 1.2)  # Boost 20%
                
                recommendations.append({
                    'id': int(product['id']) if pd.notna(product['id']) else 0,
                    'productDisplayName': product_name,
                    'articleType': str(product['articleType']) if pd.notna(product['articleType']) else 'Unknown',
                    'baseColour': str(product['baseColour']) if pd.notna(product['baseColour']) else 'Unknown',
                    'season': str(product.get('season', '')) if pd.notna(product.get('season', '')) else '',
                    'usage': str(product.get('usage', '')) if pd.notna(product.get('usage', '')) else '',
                    'similarity_score': round(base_score, 2),
                    'exact_match': bool(base_score > 75),
                    'has_image': True,
                    'rank': idx + 1,
                    'match_type': 'text_similarity',
                    'product_category': product_category
                })
            
            print(f"✅ Found {len(recommendations)} text matches")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error in text recommendation: {e}")
            return {"error": f"Text search failed: {str(e)}"}

class HybridRecommender:
    """Hybrid recommender yang optimized"""
    
    def __init__(self, text_recommender, image_recommender, df):
        self.text_recommender = text_recommender
        self.image_recommender = image_recommender
        self.df = df
    
    def recommend_by_text_query(self, query_text, n_recommendations=None):
        """Rekomendasi text"""
        try:
            return self.text_recommender.recommend_by_text(query_text, n_recommendations)
        except Exception as e:
            print(f"❌ Error in text recommendation: {e}")
            return {"error": f"Text recommendation failed: {str(e)}"}
    
    def recommend_by_image_upload(self, image_path, n_recommendations=None):
        """Rekomendasi gambar"""
        try:
            return self.image_recommender.recommend_by_image(image_path, n_recommendations)
        except Exception as e:
            print(f"❌ Error in image recommendation: {e}")
            return {"error": f"Image recommendation failed: {str(e)}"}

class RecommendationSystem:
    """Main system dengan shape-based matching"""
    
    def __init__(self):
        self.config = Config()
        self.styles_df = None
        self.shape_matcher = None
        self.text_recommender = None
        self.hybrid_recommender = None
        
    def initialize_system(self):
        """Initialize shape-based system"""
        print("🚀 Initializing SHAPE-BASED Recommendation System...")
        
        # Load data
        if not self._load_data():
            return False
        
        try:
            # Initialize recommenders
            self.text_recommender = SmartTextRecommender(self.styles_df)
            self.shape_matcher = UltraAccurateShapeMatcher(self.styles_df, self.config)
            self.hybrid_recommender = HybridRecommender(
                self.text_recommender, self.shape_matcher, self.styles_df
            )
            
            print("✅ SHAPE-BASED system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            traceback.print_exc()
            return False
    
    def _load_data(self):
        """Load dataset"""
        try:
            possible_files = [self.config.STYLES_FILE, 'styles.csv', 'fashion-dataset/styles.csv']
            data_file = None
            
            for file in possible_files:
                if os.path.exists(file):
                    data_file = file
                    break
            
            if not data_file:
                print("❌ No dataset file found!")
                return False
                
            self.styles_df = pd.read_csv(data_file, on_bad_lines='skip')
            print(f"✅ Dataset loaded: {self.styles_df.shape[0]} products")
            
            # Basic cleaning
            self.styles_df = self.styles_df.drop_duplicates(subset=['id'])
            text_cols = ['productDisplayName', 'articleType', 'baseColour']
            for col in text_cols:
                if col in self.styles_df.columns:
                    self.styles_df[col] = self.styles_df[col].fillna('Unknown')
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def get_dataset_statistics(self):
        """Get dataset statistics"""
        if self.styles_df is None:
            return {}
        
        return {
            'total_products': int(len(self.styles_df)),
            'unique_types': int(self.styles_df['articleType'].nunique()),
            'unique_colors': int(self.styles_df['baseColour'].nunique()),
        }

if __name__ == "__main__":
    system = RecommendationSystem()
    system.initialize_system()