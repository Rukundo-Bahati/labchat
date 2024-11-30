import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up the model configuration
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model object
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are an expert in teaching sciences to kids. Your task is to engage in conversations about science and answer questions. Explain scientific concepts so that they are easily understandable. Use analogies and examples that are relatable. Use humor and make the conversation an educational experience. Suggest ways that those concepts can be related to the real world with observations and experiments."
)

app = Flask(__name__)

# Store chat history in memory
chat_history = []

def get_model_response(user_input, history):
    history.append({"role": "user", "parts": [user_input]})

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
    except AttributeError:
        return "Error: Method issue"

    history.append({"role": "model", "parts": [response.text]})
    return response.text

# Flask route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle user input and get model response
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    model_response = get_model_response(user_input, chat_history)

    return jsonify({
        'user_input': user_input,
        'model_response': model_response
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
