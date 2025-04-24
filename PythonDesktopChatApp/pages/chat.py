import customtkinter as ctk
import datetime
import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot_model.pth')
DIMENSIONS_PATH = os.path.join(BASE_DIR, 'dimensions.json')

nltk.download('punkt')
nltk.download('wordnet')

class ChatBotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class ChatBotAssistant:

    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    def bag_of_words(self, words):
        bag = [0] * len(self.vocabulary)
        for word in words:
            if word in self.vocabulary:
                bag[self.vocabulary.index(word)] = 1
        return bag

    def parse_intents(self):
        print("Parsing intents...")
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))
            print(f"Found {len(self.documents)} documents")
            print(f"Vocabulary size: {len(self.vocabulary)}")
            print(f"Intents: {self.intents}")
        else:
            print(f"Error: {self.intents_path} not found")

    def prepare_data(self):
        print("Preparing data...")
        if not self.documents:
            print("No documents found. Please parse intents first.")
            return

        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

    def train_model(self, batch_size, lr, epochs):
        if self.X is None or self.y is None:
            print("Error: Data not prepared. Call prepare_data() first.")
            return

        print("Starting training...")
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        if self.intents_responses.get(predicted_intent):
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None

def get_stocks():
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']

    if len(stocks) >= 3:
        print(random.sample(stocks, 3))
    else:
        print(stocks)

trainModel = True

def train_model_if_needed():
    print("Initializing assistant...")
    print(f"Looking for intents file at: {INTENTS_PATH}")
    assistant = ChatBotAssistant(INTENTS_PATH, function_mappings={'stocks': get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    
    if trainModel:
        print("Training new model...")
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model(MODEL_PATH, DIMENSIONS_PATH)
    else:
        print("Loading existing model...")
        assistant.load_model(MODEL_PATH, DIMENSIONS_PATH)
    
    return assistant

class ChatPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.assistant = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.chat_container = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.chat_container.grid_rowconfigure(0, weight=1)
        self.chat_container.grid_columnconfigure(0, weight=1)

        self.chat_display_frame = ctk.CTkFrame(self.chat_container)
        self.chat_display_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        self.chat_display_frame.grid_rowconfigure(0, weight=1)
        self.chat_display_frame.grid_columnconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(
            self.chat_display_frame,
            wrap="word",
            state="disabled",
            font=("Roboto", 14),
            corner_radius=15,
            border_width=0,
            fg_color=("#F0F0F0", "#2B2B2B")
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.input_frame = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.user_input = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Type your message...",
            height=50,
            font=("Roboto", 14),
            corner_radius=10,
            border_width=0
        )
        self.user_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            command=self.send_message,
            height=50,
            font=("Roboto", 14, "bold"),
            corner_radius=10
        )
        self.send_button.grid(row=0, column=1)

        home_button = ctk.CTkButton(
            self.chat_container,
            text="← Home",
            command=lambda: controller.show_page("HomePage"),
            height=40,
            font=("Roboto", 14),
            corner_radius=10,
            fg_color="transparent",
            border_width=2
        )
        home_button.grid(row=2, column=0, pady=(20, 0), sticky="e")

        self.assistant = train_model_if_needed()

    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if message:
            self.update_chat("You", message)
            self.user_input.delete(0, 'end')
            response = self.assistant.process_message(message)
            if response:
                self.update_chat("Bot", response)
            else:
                self.update_chat("Bot", "I'm not sure how to respond to that. Could you try rephrasing?")

    def update_chat(self, sender, message):
        self.chat_display.configure(state="normal")
        timestamp = datetime.datetime.now().strftime('%H:%M')
        
        if sender == "You":
            sender_color = "#1E88E5"  
        else:
            sender_color = "#43A047" 
            
        self.chat_display.insert("end", f"{sender} [{timestamp}]\n", "sender")
        self.chat_display.insert("end", f"{message}\n\n", "message")
        
        self.chat_display.tag_config("sender", foreground=sender_color)
        self.chat_display.tag_config("message", foreground="#2B2B2B")  
        
        if ctk.get_appearance_mode() == "Dark":
            self.chat_display.tag_config("message", foreground="#FFFFFF") 
        
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
