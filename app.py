import chatbot 
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
import base64
# from chatbot_module import get_gemini_response  # Original incorrect import
import google.generativeai as genai

app = Flask(__name__)

'''
# Chatbot route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    response = None
    error = None
    if request.method == 'POST':
        user_query = request.form['query']
        try:
            response = chatbot_module.get_gemini_response(user_query)
        except Exception as e:
            error = str(e)
            print(f"Chatbot Error: {e}")
    return render_template('chatbot.html', response=response, error=error)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        user_query = data.get('message')
        if not user_query:
            return jsonify({'error': 'Missing "message" in request'}), 400

        response = chatbot_module.get_gemini_response(user_query)
        return jsonify({'response': response})
    except Exception as e:
        print(f"API Chatbot Error: {e}")
        return jsonify({'error': str(e)}), 500
'''
'''
*********************
'''
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_route():  # renamed to avoid conflict
    response = None
    error = None
    if request.method == 'POST':
        user_query = request.form['query']
        try:
            response = chatbot.get_gemini_response(user_query)  # fixed usage
        except Exception as e:
            error = str(e)
            print(f"Chatbot Error: {e}")
    return render_template('chatbot.html', response=response, error=error)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        user_query = data.get('message')
        if not user_query:
            return jsonify({'error': 'Missing "message" in request'}), 400

        response = chatbot.get_gemini_response(user_query)  # fixed usage
        return jsonify({'response': response})
    except Exception as e:
        print(f"API Chatbot Error: {e}")
        return jsonify({'error': str(e)}), 500

'''
*********************
'''
# Load the pre-trained CNN model
MODEL_PATH = 'C://Users//Microsoft//Desktop//DR DETECTION//diabetic_retinopathy_model.h5'  # Update this path
model = load_model(MODEL_PATH)

# Class names for diabetic retinopathy classification
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        # Generate graph for severity awareness
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, predictions[0], color=['#A7C7E7', '#F4BFBF', '#FFD580', '#FF6961', '#9B59B6'])
        ax.set_title('Diabetic Retinopathy Severity')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Prediction Probability')

        # Convert plot to image and encode as base64
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        graph_url = base64.b64encode(img_io.getvalue()).decode()

        # Define precautionary tips for each class
        precaution_tips = {
            'No_DR': 'Maintain a healthy lifestyle and regular check-ups.',
            'Mild': 'Monitor blood sugar levels and consider dietary changes.',
            'Moderate': 'Consult an ophthalmologist and manage blood sugar more strictly.',
            'Severe': 'Seek immediate medical care and monitor vision changes closely.',
            'Proliferate_DR': 'Intensive treatment and possible surgery may be required.'
        }

        return render_template('result.html', predicted_class=predicted_class, graph_url=graph_url, tips=precaution_tips[predicted_class])

    return render_template('predict.html')

# Contact page route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Added missing diabetic_detection route
@app.route('/diabetic_detection')
def diabetic_detection():
    return render_template('diabetic_detection.html')

if __name__ == '__main__':
    app.run(debug=True)