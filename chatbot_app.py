import os
from flask import Flask
from flask import Flask, render_template, request, jsonify
import logging
from flask_cors import CORS
import requests  # Ensure this is imported

app = Flask(__name__)
CORS(app)  # Handle CORS issues

API_URL = "https://chatbot-rag.onrender.com/ask_question"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def distributed_computing_chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    logging.debug('Chat endpoint hit')
    user_input = request.json.get('message')
    logging.debug(f'Received message: {user_input}')
    # result = user_input + " Thanks for typing this"
    # return jsonify({'response': result})

    # Make an API call with the user input
    try:
        # Send a GET request with query parameters
        response = requests.get(API_URL, params={'question': user_input, 'verbose': 'false'})
        response.raise_for_status()
        api_result = response.text  # Use .text for plain text response
        logging.debug(f'API response: {api_result}')
        result = api_result  # No need for .get() as we're expecting a plain text response
    except requests.RequestException as e:
        logging.error(f'API call failed: {e}')
        result = 'Error processing your request'

    return jsonify({'response': result})

if __name__ == '__main__':
    # Use the PORT environment variable, defaulting to 5000 if it's not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

