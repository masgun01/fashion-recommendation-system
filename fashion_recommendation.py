import pandas as pd
import numpy as np
import os
import warnings
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import traceback
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Config:
    IMAGE_DIR = 'dataset/images'  # Changed to match your structure
    STYLES_FILE = 'dataset/styles.csv'  # Changed path
    SHAPE_CACHE_FILE = 'shape_features_cache.pkl'

class OptimizedShapeFeatureExtractor:
    """Optimized shape feature extractor untuk free tier"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def extract_shape_features(self, image_path):
        """Extract shape features dengan memory optimization"""
        try:
            # Load image dengan optimasi memory
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load langsung grayscale
            if img is None:
                return None
            
            # Resize lebih kecil untuk free tier
            img = cv2.resize(img, (150, 150))  # Reduced from 200x200
            
            # Simple blur
            blurred = cv2.GaussianBlur(img, (3, 3), 0)  # Reduced kernel
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return self._get_basic_features(img)  # Fallback
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Extract hanya features penting untuk hemat memory
            features = self._extract_essential_features(largest_contour, img)
            
            # Clean up memory
            del img, blurred, edges, contours
            gc.collect()
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting shape features: {e}")
            return None
    
    def _extract_essential_features(self, contour, img):
        """Extract hanya features yang paling penting"""
        features = {}
        
        # Basic geometric features saja
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        features['aspect_ratio'] = w / h if h > 0 else 0
        features['area_ratio'] = area / (w * h) if w * h > 0 else 0
        features['compactness'] = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # Simplified Hu moments (hanya 3 pertama)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        for i in range(3):  # Hanya 3 Hu moments
            if hu_moments[i] != 0:
                features[f'hu_{i}'] = -np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
            else:
                features[f'hu_{i}'] = 0
        
        return features
    
    def _get_basic_features(self, img):
        """Fallback features jika contour tidak ditemukan"""
        features = {
            'aspect_ratio': 1.0,
            'area_ratio': 0.5,
            'compactness': 1.0,
            'hu_0': 0.1,
            'hu_1': 0.1,
            'hu_2': 0.1
        }
        return features

class ProductCategorizer:
    """Klasifikasi kategori produk berdasarkan nama"""
    
    def __init__(self):
        self.category_keywords = {
            'shoes': ['shoe', 'sneaker', 'boot', 'footwear', 'loafer'],
            'sandals': ['sandal', 'flipflop', 'slide'],
            'dresses': ['dress', 'gown', 'frock'],
            'shirts': ['shirt', 'top', 'blouse', 't-shirt'],
            'pants': ['pant', 'trouser', 'jeans'],
            'bags': ['bag', 'purse', 'backpack'],
            'accessories': ['hat', 'cap', 'belt', 'scarf'],
            'jackets': ['jacket', 'coat', 'blazer'],
            'skirts': ['skirt', 'mini', 'midi'],
            'shorts': ['short', 'bermuda']
        }
    
    def predict_category(self, product_name=""):
        if not product_name:
            return 'unknown'
        
        product_name_lower = product_name.lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in product_name_lower:
                    return category
        
        return 'unknown'

class UltraAccurateShapeMatcher:
    """Shape matcher yang dioptimasi untuk free tier"""
    
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.feature_extractor = OptimizedShapeFeatureExtractor()
        self.categorizer = ProductCategorizer()
        self.features_cache = {}
        self.image_mapping = {}
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize dengan optimasi memory"""
        print("🔄 Initializing OPTIMIZED Shape-Based System...")
        
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
                self._build_features_cache_optimized()
        else:
            self._build_features_cache_optimized()
    
    def _build_features_cache_optimized(self):
        """Build cache dengan optimasi memory"""
        print("🔄 Building optimized shape features cache...")
        
        self._build_image_mapping_optimized()
        
        # Batasi jumlah gambar untuk free tier (max 1000)
        max_images = 1000
        valid_count = 0
        
        items_to_process = list(self.image_mapping.items())[:max_images]
        
        for product_id, filename in tqdm(items_to_process, desc="Extracting shapes"):
            image_path = os.path.join(self.config.IMAGE_DIR, filename)
            
            if os.path.exists(image_path):
                shape_features = self.feature_extractor.extract_shape_features(image_path)
                
                if shape_features:
                    product_row = self.df[self.df['id'].astype(str) == product_id]
                    product_name = ""
                    if not product_row.empty:
                        product_name = product_row.iloc[0].get('productDisplayName', '')
                    
                    category = self.categorizer.predict_category(product_name)
                    
                    self.features_cache[product_id] = {
                        'shape_features': shape_features,
                        'category': category,
                        'product_name': product_name
                    }
                    valid_count += 1
            
            # Periodic garbage collection
            if valid_count % 100 == 0:
                gc.collect()
        
        # Save cache
        cache_data = {
            'features_cache': self.features_cache,
            'image_mapping': self.image_mapping
        }
        with open(self.config.SHAPE_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ Built optimized cache with {valid_count} shape features")
        self._print_category_stats()
    
    def _build_image_mapping_optimized(self):
        """Build image mapping dengan optimasi"""
        if not os.path.exists(self.config.IMAGE_DIR):
            print("⚠️ Images directory not found, creating dummy mapping...")
            # Create dummy mapping dari dataset
            for _, row in self.df.head(1000).iterrows():  # Max 1000
                self.image_mapping[str(row['id'])] = f"{row['id']}.jpg"
            return
        
        try:
            image_files = [f for f in os.listdir(self.config.IMAGE_DIR) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:2000]  # Limit files
            
            for file in image_files[:1000]:  # Max 1000 images
                name_without_ext = os.path.splitext(file)[0]
                self.image_mapping[name_without_ext] = file
            
            print(f"✅ Mapped {len(self.image_mapping)} products to images")
        except Exception as e:
            print(f"❌ Error building image mapping: {e}")
            # Fallback: create from dataset
            for _, row in self.df.head(500).iterrows():
                self.image_mapping[str(row['id'])] = f"{row['id']}.jpg"
    
    def _print_category_stats(self):
        categories = {}
        for data in self.features_cache.values():
            cat = data['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("📊 Category Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {cat}: {count} products")
    
    def _calculate_shape_similarity(self, features1, features2):
        """Hitung similarity yang dioptimasi"""
        if not features1 or not features2:
            return 0.0
        
        try:
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 0.0
            
            vec1 = np.array([features1[k] for k in common_keys])
            vec2 = np.array([features2[k] for k in common_keys])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            return 0.0
    
    def recommend_by_image(self, image_path, n_recommendations=None):
        """Rekomendasi gambar yang dioptimasi"""
        try:
            print("🎯 OPTIMIZED SHAPE-BASED MATCHING...")
            
            uploaded_features = self.feature_extractor.extract_shape_features(image_path)
            if not uploaded_features:
                return {"error": "Cannot extract shape features from uploaded image"}
            
            similar_products = []
            
            # Batasi processing untuk free tier
            max_products_to_check = 500
            items_to_check = list(self.features_cache.items())[:max_products_to_check]
            
            for product_id, cache_data in items_to_check:
                product_features = cache_data['shape_features']
                similarity = self._calculate_shape_similarity(uploaded_features, product_features)
                
                if similarity > 0.2:  # Lower threshold
                    similar_products.append({
                        'product_id': product_id,
                        'similarity': similarity,
                        'category': cache_data['category']
                    })
            
            similar_products.sort(key=lambda x: x['similarity'], reverse=True)
            
            if n_recommendations is None:
                n_recommendations = min(20, len(similar_products))  # Less results
            
            recommendations = []
            for i, item in enumerate(similar_products[:n_recommendations]):
                product_id = item['product_id']
                similarity_score = float(item['similarity'] * 100)
                
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
            
            # Clean memory
            gc.collect()
            
            return recommendations if recommendations else {"error": "No similar products found"}
            
        except Exception as e:
            error_msg = f"Error in shape matching: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

class SmartTextRecommender:
    """Text recommender yang dioptimasi"""
    
    def __init__(self, df):
        self.df = df
        self.categorizer = ProductCategorizer()
        self._build_text_similarity_matrix_optimized()
    
    def _build_text_similarity_matrix_optimized(self):
        """Build text similarity dengan optimasi memory"""
        print("🔄 Building optimized text similarity matrix...")
        
        try:
            # Gunakan sample yang lebih kecil untuk free tier
            sample_size = min(2000, len(self.df))
            self.df_sample = self.df.head(sample_size).copy()
            
            self.df_sample['search_features'] = (
                self.df_sample['productDisplayName'].fillna('') + ' ' +
                self.df_sample['articleType'].fillna('') + ' ' +
                self.df_sample['baseColour'].fillna('')
            )
            
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=3000,  # Reduced features
                min_df=1
            )
            
            self.tfidf_matrix = self.tfidf.fit_transform(self.df_sample['search_features'])
            print(f"✅ Optimized text matrix: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            print(f"❌ Error building text similarity: {e}")
            raise
    
    def recommend_by_text(self, query_text, n_recommendations=None):
        """Rekomendasi text yang dioptimasi"""
        try:
            print(f"🔍 Optimized text search: '{query_text}'")
            
            query_vec = self.tfidf.transform([query_text.lower()])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            relevant_indices = np.where(similarity_scores > 0.01)[0]
            
            if not relevant_indices.size:
                return {"error": f"No products found for '{query_text}'"}
            
            sorted_indices = relevant_indices[np.argsort(similarity_scores[relevant_indices])[::-1]]
            
            if n_recommendations is None:
                n_recommendations = min(20, len(sorted_indices))  # Less results
            
            results = self.df_sample.iloc[sorted_indices[:n_recommendations]]
            similarity_values = similarity_scores[sorted_indices[:n_recommendations]]
            
            recommendations = []
            for idx, (_, product) in enumerate(results.iterrows()):
                product_name = str(product['productDisplayName']) if pd.notna(product['productDisplayName']) else ''
                product_category = self.categorizer.predict_category(product_name)
                
                base_score = float(similarity_values[idx] * 100)
                
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
    """Hybrid recommender yang dioptimasi"""
    
    def __init__(self, text_recommender, image_recommender, df):
        self.text_recommender = text_recommender
        self.image_recommender = image_recommender
        self.df = df
    
    def recommend_by_text_query(self, query_text, n_recommendations=None):
        try:
            return self.text_recommender.recommend_by_text(query_text, n_recommendations)
        except Exception as e:
            return {"error": f"Text recommendation failed: {str(e)}"}
    
    def recommend_by_image_upload(self, image_path, n_recommendations=None):
        try:
            return self.image_recommender.recommend_by_image(image_path, n_recommendations)
        except Exception as e:
            return {"error": f"Image recommendation failed: {str(e)}"}

class RecommendationSystem:
    """Main system yang dioptimasi untuk free tier"""
    
    def __init__(self):
        self.config = Config()
        self.styles_df = None
        self.shape_matcher = None
        self.text_recommender = None
        self.hybrid_recommender = None
        
    def initialize_system(self):
        """Initialize system dengan optimasi"""
        print("🚀 Initializing OPTIMIZED Recommendation System...")
        
        if not self._load_data_optimized():
            return False
        
        try:
            # Initialize dengan delay untuk hemat memory
            import time
            time.sleep(1)
            
            self.text_recommender = SmartTextRecommender(self.styles_df)
            time.sleep(1)
            
            self.shape_matcher = UltraAccurateShapeMatcher(self.styles_df, self.config)
            time.sleep(1)
            
            self.hybrid_recommender = HybridRecommender(
                self.text_recommender, self.shape_matcher, self.styles_df
            )
            
            print("✅ OPTIMIZED system initialized successfully!")
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False
    
    def _load_data_optimized(self):
        """Load dataset dengan optimasi"""
        try:
            possible_files = [
                'dataset/styles.csv',  # Your structure
                'styles.csv', 
                'fashion-dataset/styles.csv'
            ]
            data_file = None
            
            for file in possible_files:
                if os.path.exists(file):
                    data_file = file
                    break
            
            if not data_file:
                print("❌ No dataset file found!")
                return False
                
            # Load hanya sample untuk free tier
            self.styles_df = pd.read_csv(data_file, on_bad_lines='skip', nrows=3000)  # Limit rows
            
            # Basic cleaning
            self.styles_df = self.styles_df.drop_duplicates(subset=['id'])
            text_cols = ['productDisplayName', 'articleType', 'baseColour']
            for col in text_cols:
                if col in self.styles_df.columns:
                    self.styles_df[col] = self.styles_df[col].fillna('Unknown')
            
            print(f"✅ Optimized dataset loaded: {self.styles_df.shape[0]} products")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def get_dataset_statistics(self):
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
