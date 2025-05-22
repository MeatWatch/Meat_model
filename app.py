from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import logging
from dotenv import load_dotenv
from database import get_db_connection, close_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meatonly')

# Load environment variables
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = 'models/meat_model.h5'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# Load TensorFlow model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("TensorFlow model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TensorFlow model: {e}")
    model = None

def predict_image(image_path):
    """
    Process an image and return prediction results.
    """
    if model is None:
        raise ValueError("Model not loaded. Cannot make predictions.")
    
    try:
        # Open, resize and preprocess the image
        img = Image.open(image_path).resize((224, 224))
        # img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0]
        
        # Process results
        confidence = float(np.max(prediction))
        label = 'fresh' if np.argmax(prediction) == 0 else 'not_fresh'
        
        return label, confidence
    except Exception as e:
        logger.error(f"Error during image prediction: {e}")
        raise

@app.route('/classify_beef', methods=['POST'])
def classify_beef():
    """
    API endpoint to classify beef images.
    """
    logger.info("Received classification request")
    
    # Check if image is in request
    if 'image' not in request.files:
        logger.warning("No image part in the request")
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        logger.warning("No image selected for uploading")
        return jsonify({'error': 'No image selected for uploading'}), 400
    
    # Check if user_id is provided
    user_id = request.form.get('user_id')
    if not user_id:
        logger.warning("User ID is required")
        return jsonify({'error': 'User ID is required'}), 400
    
    # Generate unique filename and save the uploaded image
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        logger.info(f"Image saved to {filepath}")
    except Exception as e:
        logger. error(f"Failed to save uploaded file: {e}")
        return jsonify({'error': 'Failed to save image file'}), 500
    
    # Make prediction
    try:
        result, confidence = predict_image(filepath)
        logger.info(f"Prediction result: {result} with confidence {confidence}")
    except Exception as e:
        logger.error(f"Failed to predict image: {e}")
        # Clean up uploaded file if prediction fails
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'Failed to process image prediction'}), 500
    
    # Save result to database
    conn = None
    cursor = None
    try:
        logger.info("Connecting to database...")
        conn = get_db_connection()
        
        if conn is None:
            logger.error("Failed to connect to database")
            # Clean up uploaded file if database connection fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Failed to connect to database for logging'}), 500
        
        cursor = conn.cursor()
        logger.info("Database cursor created")
        
        # Insert classification result
        sql = """
            INSERT INTO classifications (user_id, meat_type, image_path, result, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """
        val = (user_id, 'beef', filepath, result, confidence)
        
        logger.info(f"Executing SQL: {sql} with values {val}")
        cursor.execute(sql, val)
        conn.commit()
        
        classification_id = cursor.lastrowid
        logger.info(f"Classification saved with ID: {classification_id}")
        
        # Clean up file after successful database operation
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted uploaded file: {filepath}")
        
        return jsonify({
            'classification_id': classification_id,
            'result': result,
            'confidence_score': confidence
        })
        
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        if conn:
            conn.rollback()
        return jsonify({'error': f'Database operation failed: {str(e)}'}), 500
        
    finally:
        close_connection(conn, cursor)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    # Test database connection
    db_status = "OK"
    try:
        conn = get_db_connection()
        if conn and conn.is_connected():
            close_connection(conn)
        else:
            db_status = "Failed to connect"
    except Exception as e:
        db_status = f"Error: {str(e)}"
    
    # Test model status
    model_status = "OK" if model is not None else "Not loaded"
    
    return jsonify({
        'status': 'up',
        'database': db_status,
        'model': model_status
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting application on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)