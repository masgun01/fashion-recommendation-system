from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io
from fashion_recommendation import RecommendationSystem
import logging
import traceback
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fashion-recommendation-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize recommendation system
recommendation_system = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_recommendation_system():
    global recommendation_system
    try:
        logger.info("🚀 Initializing SHAPE-BASED Recommendation System...")
        recommendation_system = RecommendationSystem()
        success = recommendation_system.initialize_system()
        if success:
            logger.info("✅ SHAPE-BASED system initialized!")
        else:
            logger.error("❌ Failed to initialize system")
        return success
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return False

def get_product_image_path(product_id):
    """Get product image path - FIXED VERSION"""
    image_dirs = ['fashion-dataset/images', 'images']
    product_id_str = str(product_id)
    
    for image_dir in image_dirs:
        if os.path.exists(image_dir):
            # Check for different image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                image_path = os.path.join(image_dir, f"{product_id_str}{ext}")
                if os.path.exists(image_path):
                    return image_path
            
            # Check for exact filename match
            try:
                for filename in os.listdir(image_dir):
                    name_without_ext = os.path.splitext(filename)[0]
                    if name_without_ext == product_id_str:
                        return os.path.join(image_dir, filename)
            except Exception as e:
                print(f"⚠️ Error reading directory {image_dir}: {e}")
                continue
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<product_id>')
def get_product_image(product_id):
    try:
        image_path = get_product_image_path(product_id)
        
        if image_path and os.path.exists(image_path):
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            return send_from_directory(directory, filename)
        else:
            return send_from_directory('static', 'placeholder.jpg')
    
    except Exception as e:
        logger.error(f"Error serving image for product {product_id}: {e}")
        return send_from_directory('static', 'placeholder.jpg')

@app.route('/recommend/text', methods=['POST'])
def recommend_by_text():
    try:
        start_time = time.time()
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"🔍 Text query: '{query}'")
        
        if recommendation_system is None:
            return jsonify({'error': 'System not initialized'}), 500
        
        recommendations = recommendation_system.hybrid_recommender.recommend_by_text_query(query, None)
        
        processing_time = time.time() - start_time
        
        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify({'error': recommendations['error']}), 400
        
        if not isinstance(recommendations, list):
            return jsonify({'error': 'Unexpected result format'}), 500
        
        return jsonify({
            'query_text': query,
            'recommendations': recommendations,
            'total_results': len(recommendations),
            'processing_time': f"{processing_time:.2f}s"
        })
    
    except Exception as e:
        logger.error(f"❌ Text recommendation error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/recommend/image', methods=['POST'])
def recommend_by_image():
    try:
        start_time = time.time()
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            try:
                logger.info(f"🔍 Processing image: {filename}")
                
                if recommendation_system is None:
                    return jsonify({'error': 'System not initialized'}), 500
                
                # SHAPE-BASED recommendation
                recommendations = recommendation_system.hybrid_recommender.recommend_by_image_upload(filepath, None)
                
                processing_time = time.time() - start_time
                
                if isinstance(recommendations, dict) and 'error' in recommendations:
                    return jsonify({'error': recommendations['error']}), 400
                
                if not isinstance(recommendations, list):
                    return jsonify({'error': 'Unexpected result format'}), 500
                
                # Convert to base64 for display
                with open(filepath, "rb") as img_file:
                    uploaded_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                response_data = {
                    'query_type': 'image',
                    'uploaded_image': f"data:image/jpeg;base64,{uploaded_image_b64}",
                    'recommendations': recommendations,
                    'total_results': len(recommendations),
                    'processing_time': f"{processing_time:.2f}s"
                }
                
                logger.info(f"✅ Shape matching completed in {processing_time:.2f}s - Found {len(recommendations)} items")
                return jsonify(response_data)
                
            finally:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass
        
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logger.error(f"❌ Image recommendation error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        if recommendation_system is None:
            return jsonify({'error': 'System not initialized'}), 500
            
        stats = recommendation_system.get_dataset_statistics()
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"❌ Statistics error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/products/random', methods=['GET'])
def get_random_products():
    try:
        n = int(request.args.get('n', 8))
        
        if recommendation_system is None:
            return jsonify({'error': 'System not initialized'}), 500
        
        if len(recommendation_system.styles_df) > 0:
            sample_df = recommendation_system.styles_df.sample(min(n, len(recommendation_system.styles_df)))
            products = []
            
            for _, product in sample_df.iterrows():
                has_image = get_product_image_path(product['id']) is not None
                products.append({
                    'id': product['id'],
                    'productDisplayName': product['productDisplayName'],
                    'articleType': product['articleType'],
                    'baseColour': product['baseColour'],
                    'season': product.get('season', ''),
                    'has_image': has_image
                })
            
            return jsonify({'products': products})
        else:
            return jsonify({'products': []})
    
    except Exception as e:
        logger.error(f"❌ Random products error: {e}")
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting SHAPE-BASED Fashion Recommendation System...")
    if initialize_recommendation_system():
        print("✅ System initialized successfully!")
        
        os.makedirs('static/uploads', exist_ok=True)
        
        if not os.path.exists('static/placeholder.jpg'):
            img = Image.new('RGB', (200, 200), color='lightgray')
            img.save('static/placeholder.jpg')
        
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start server")