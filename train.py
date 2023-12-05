import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents from JSON file
def load_intents(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Preprocess intents data
def preprocess_intents(intents):
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return all_words, tags, xy

# Create training data
def create_training_data(all_words, tags, xy):
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    return np.array(X_train), np.array(y_train)

# Create dataset
def create_dataset(X_train, y_train):
    class ChatDataset(Dataset):
        def __init__(self, X, y):
            self.n_samples = len(X)
            self.x_data = X
            self.y_data = y

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    return ChatDataset(X_train, y_train)

# Train the model
def train_model(model, train_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, loss.item()

# Save the trained model
def save_model(model, input_size, hidden_size, output_size, all_words, tags):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'Training complete. File saved to {FILE}')

def main():
    intents = load_intents('intents.json')
    all_words, tags, xy = preprocess_intents(intents)
    X_train, y_train = create_training_data(all_words, tags, xy)
    input_size = len(X_train[0])
    output_size = len(tags)

    dataset = create_dataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

    model = NeuralNet(input_size, hidden_size=8, output_size=output_size)
    trained_model, final_loss = train_model(model, train_loader, num_epochs=1000)

    save_model(trained_model, input_size, hidden_size=8, output_size=output_size, all_words=all_words, tags=tags)

if __name__ == "__main__":
    main()
