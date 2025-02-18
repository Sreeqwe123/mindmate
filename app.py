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
data = torch.load(FILE)
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
genai.configure(api_key=os.environ["AIzaSyCZUjrMRQf8rRsqizwn1a7UiAZslE0r-9U"])

generation_config = {
    "temperature": 1.95,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="It is a mental health support chatbot named MindMate. Do not respond to non-mental health-related questions."
)

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.json.get("user_id")  # Unique user identifier
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Initialize chat session if not exists
    if user_id not in chat_sessions:
        chat_sessions[user_id] = []

    chat_sessions[user_id].append(f"You: {user_message}")  # Store user input

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
    else:
        # Use Gemini AI for generating a mental health-related response
        chat_session = gemini_model.start_chat(history=[])
        response_data = chat_session.send_message(user_message)
        response_text = response_data.text if response_data else "I'm here to support you. How are you feeling today?"

        chat_sessions[user_id].append(f"{bot_name}: {response_text}")
        return jsonify({"bot": response_text, "context": chat_sessions[user_id][-5:]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
