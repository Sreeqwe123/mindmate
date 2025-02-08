import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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
chat_history = []  # List to store conversation history

print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    chat_history.append(f"You: {sentence}")

    sentence = tokenize(sentence)
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
                bot_response = random.choice(intent["responses"])
                chat_history.append(f"{bot_name}: {bot_response}")
                print(f"{bot_name}: {bot_response}")
                break
    else:
        bot_response = "I do not understand..."
        chat_history.append(f"{bot_name}: {bot_response}")
        print(f"{bot_name}: {bot_response}")

    # Print the last 5 messages to simulate a contextual conversation
    print("\nChat History:")
    for message in chat_history[-5:]:  # Display only last 5 messages
        print(message)
    print("-" * 30)
