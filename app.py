from flask import Flask, request, jsonify
import torch
import random
import json
import os
import google.generativeai as genai
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents
with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE, map_location=device)  # Ensures compatibility across devices
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "MindMate"
chat_sessions = {}  # Store chat history

# Configure Google Gemini API
api_key = os.getenv("GEMINI_API_KEY")  # Use os.getenv() to avoid KeyError
if not api_key:
    raise ValueError("Missing Gemini API Key! Set 'GEMINI_API_KEY' as an environment variable.")

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1.2,  # Adjusted for better mental health responses
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 512,  # Reduced to prevent long responses
    "response_mime_type": "application/json",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="You are MindMate, a chatbot that provides mental health support. Keep responses concise and empathetic."
)

# Create a single chat session for Gemini AI to maintain context across interactions
chat_session = gemini_model.start_chat(history=[])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user_id")
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Initialize chat session if not exists
    if user_id not in chat_sessions:
        chat_sessions[user_id] = []

    chat_sessions[user_id].append(f"You: {user_message}")  # Store user input

    # Process message with the trained chatbot model
    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Use predefined responses from intents.json
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                chat_sessions[user_id].append(f"{bot_name}: {response}")  # Store bot response
                return jsonify({"bot": response, "context": chat_sessions[user_id][-5:]})  # Return last 5 messages

    # Use Gemini AI for generating a mental health-related response
    response_data = chat_session.send_message(user_message)
    response_text = getattr(response_data, "text", "I'm here for you. How are you feeling today?")

    chat_sessions[user_id].append(f"{bot_name}: {response_text}")
    return jsonify({"bot": response_text, "context": chat_sessions[user_id][-5:]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
