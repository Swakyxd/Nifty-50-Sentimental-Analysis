"""
Flask Web Application for NIFTY 50 Market Prediction
"""
from flask import Flask, render_template, request, jsonify
from predict_market import MarketPredictor
from live_news_fetcher import LiveNewsFetcher
import os
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nifty50-prediction-secret-key'

# Initialize predictor
MODEL_DIR = 'models/finbert_lstm'  # Corrected path
CHECKPOINT_PATH = 'processed/nifty50_NIFTY 50_minute_featured.csv'  # Use processed data

try:
    predictor = MarketPredictor(MODEL_DIR)
    print("✅ Predictor initialized successfully")
except Exception as e:
    print(f"❌ Error initializing predictor: {e}")
    print(f"   Model directory: {MODEL_DIR}")
    print(f"   Please ensure the model files exist")
    predictor = None

# Initialize live news fetcher
try:
    news_fetcher = LiveNewsFetcher()
    print("✅ Live News Fetcher initialized successfully")
except Exception as e:
    print(f"❌ Error initializing news fetcher: {e}")
    news_fetcher = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized. Please check the model files.'
            }), 500
        
        # Get news text and batch selection from request
        data = request.get_json()
        news_text = data.get('news', '')
        selected_batch = data.get('batch', 'best_model')
        
        if not news_text or len(news_text.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'Please provide valid news text (at least 10 characters)'
            }), 400
        
        # Determine model path based on batch selection
        if selected_batch == 'best_model':
            model_path = MODEL_DIR
        elif selected_batch.startswith('batch_'):
            batch_num = selected_batch.split('_')[1]
            batch_year_map = {
                '1': '2013-2015',
                '2': '2016-2018',
                '3': '2019-2021',
                '4': '2022-2024',
                '5': '2025-2025'
            }
            batch_folder = f"batch_{batch_num}_{batch_year_map.get(batch_num, '')}"
            batch_dir = os.path.join(MODEL_DIR, batch_folder)
            
            # Check if batch model exists
            if not os.path.exists(batch_dir):
                return jsonify({
                    'success': False,
                    'error': f'Batch model not found: {batch_dir}'
                }), 404
            
            model_path = batch_dir
        else:
            model_path = MODEL_DIR
        
        # Create a temporary predictor for the selected batch (if different)
        if model_path != MODEL_DIR:
            try:
                batch_predictor = MarketPredictor(model_path)
                result = batch_predictor.predict(CHECKPOINT_PATH, news_text)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to load batch model: {str(e)}'
                }), 500
        else:
            # Use default predictor
            result = predictor.predict(CHECKPOINT_PATH, news_text)
        
        # Format response
        response = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': {
                'direction': result['direction'],
                'confidence': f"{result['confidence']:.2f}%",
                'confidence_raw': result['confidence'],
                'current_price': f"₹{result.get('current_price', 0):.2f}",
                'predicted_price': f"₹{result.get('predicted_price', 0):.2f}",
                'price_change': f"{result.get('price_change_percent', 0):.4f}%",
                'price_change_value': f"₹{result.get('price_change_value', 0):.2f}",
                'signal_strength': result.get('signal_strength', 'MODERATE'),
                'price_source': result.get('price_source', 'N/A')
            },
            'model_info': {
                'validation_auc': f"{result.get('validation_auc', 0):.2f}%",
                'trained_epoch': result.get('trained_epoch', 'N/A')
            },
            'sentiment_analysis': {
                'sentiment_positive': result.get('sentiment_positive', 0.0),
                'sentiment_negative': result.get('sentiment_negative', 0.0),
                'sentiment_neutral': result.get('sentiment_neutral', 1.0),
                'sentiment_score': result.get('sentiment_score', 0.0),
                'sentiment_confidence': result.get('sentiment_confidence', 0.5),
                'news_count': result.get('news_count', 0)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction request"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        data = request.get_json()
        news_list = data.get('news_list', [])
        
        if not news_list or len(news_list) == 0:
            return jsonify({
                'success': False,
                'error': 'Please provide at least one news article'
            }), 400
        
        # Make batch prediction
        results = predictor.predict_batch(CHECKPOINT_PATH, news_list)
        
        # Format response
        formatted_results = []
        for idx, result in enumerate(results):
            formatted_results.append({
                'index': idx + 1,
                'direction': result['direction'],
                'confidence': f"{result['confidence']:.2f}%",
                'predicted_price': f"₹{result.get('predicted_price', 0):.2f}",
                'price_change': f"{result.get('price_change_percent', 0):.4f}%"
            })
        
        response = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_predictions': len(formatted_results),
            'predictions': formatted_results
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@app.route('/fetch-live-news')
def fetch_live_news():
    """Fetch live financial news from NewsAPI"""
    try:
        if news_fetcher is None:
            return jsonify({
                'success': False,
                'error': 'News fetcher not initialized'
            }), 500
        
        # Get parameters from query string
        hours_back = int(request.args.get('hours', 12))
        max_articles = int(request.args.get('max', 10))
        
        # Fetch combined news
        articles = news_fetcher.fetch_combined_news(max_total=max_articles)
        
        if not articles:
            return jsonify({
                'success': False,
                'error': 'No articles found',
                'articles': []
            })
        
        # Get summary
        summary = news_fetcher.get_news_summary(articles)
        
        # Format response
        response = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': summary,
            'articles': articles
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to fetch news: {str(e)}'
        }), 500

@app.route('/predict-from-live-news', methods=['POST'])
def predict_from_live_news():
    """Fetch live news and make predictions"""
    try:
        if predictor is None or news_fetcher is None:
            return jsonify({
                'success': False,
                'error': 'Predictor or news fetcher not initialized'
            }), 500
        
        # Get parameters
        data = request.get_json()
        max_articles = data.get('max_articles', 5)
        hours_back = data.get('hours_back', 12)
        
        # Fetch live news
        articles = news_fetcher.fetch_combined_news(max_total=max_articles)
        
        if not articles:
            return jsonify({
                'success': False,
                'error': 'No live news articles found'
            }), 404
        
        # Make predictions for each article
        predictions = []
        for idx, article in enumerate(articles[:max_articles]):
            try:
                # Use full_text for prediction
                news_text = article['full_text']
                
                if len(news_text) >= 10:
                    result = predictor.predict(CHECKPOINT_PATH, news_text)
                    
                    predictions.append({
                        'article_index': idx + 1,
                        'title': article['title'][:100] + '...' if len(article['title']) > 100 else article['title'],
                        'source': article['source'],
                        'published_at': article['published_at'],
                        'url': article['url'],
                        'prediction': {
                            'direction': result['direction'],
                            'confidence': f"{result['confidence']:.2f}%",
                            'confidence_raw': result['confidence'],
                            'predicted_price': f"₹{result.get('predicted_price', 0):.2f}",
                            'price_change': f"{result.get('price_change_percent', 0):.4f}%",
                            'signal_strength': result.get('signal_strength', 'MODERATE')
                        }
                    })
            except Exception as e:
                print(f"⚠️ Error predicting article {idx+1}: {e}")
                continue
        
        # Calculate aggregate prediction
        if predictions:
            up_count = sum(1 for p in predictions if p['prediction']['direction'] == 'UP')
            down_count = len(predictions) - up_count
            avg_confidence = sum(p['prediction']['confidence_raw'] for p in predictions) / len(predictions)
            
            overall_direction = 'UP' if up_count > down_count else 'DOWN'
            overall_confidence = (max(up_count, down_count) / len(predictions)) * 100
            
            aggregate = {
                'direction': overall_direction,
                'confidence': f"{overall_confidence:.2f}%",
                'up_count': up_count,
                'down_count': down_count,
                'total_articles': len(predictions),
                'avg_confidence': f"{avg_confidence:.2f}%"
            }
        else:
            aggregate = None
        
        # Format response
        response = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'news_summary': news_fetcher.get_news_summary(articles),
            'aggregate_prediction': aggregate,
            'individual_predictions': predictions
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Live news prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None,
        'news_fetcher_loaded': news_fetcher is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting NIFTY 50 Market Prediction Web App")
    print("="*80)
    print(f"📊 Model Directory: {MODEL_DIR}")
    print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
    print(f"🌐 Access the app at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
