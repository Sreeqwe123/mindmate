from flask import Flask, request, jsonify
import torch
import random
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

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

bot_name = "Sam"
chat_sessions = {}  # Dictionary to store user chat history

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
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                chat_sessions[user_id].append(f"{bot_name}: {response}")  # Store bot response
                return jsonify({"bot": response, "context": chat_sessions[user_id][-5:]})  # Return last 5 messages
    else:
        response = "I do not understand..."
        chat_sessions[user_id].append(f"{bot_name}: {response}")
        return jsonify({"bot": response, "context": chat_sessions[user_id][-5:]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
