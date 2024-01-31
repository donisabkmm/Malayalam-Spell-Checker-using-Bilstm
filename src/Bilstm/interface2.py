import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data from CSV
filename = open('../../DataSet/Final_dataset.csv', 'r', encoding='utf-8')
data = csv.reader(filename)
words = [word for row in data for word in row]

# Create character to index mapping
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(set(''.join(words))))}

# Prepare input data
max_len = max(len(word) for word in words)
X = [[char_to_idx[char] for char in word] for word in words]
X = [[x + [0] * (max_len - len(x))] for x in X]
X = torch.tensor(X, dtype=torch.long)

# Define BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        output = self.dropout(output)
        output = self.fc(output)
        return output

# Hyperparameters
vocab_size = len(char_to_idx) + 1
embedding_dim = 50
hidden_dim = 64
num_epochs = 50  # Increased for better training
batch_size = 32

# Train-Validation split
split = int(0.8 * len(X))
train_data, val_data = X[:split], X[split:]

# Instantiate model, criterion, and optimizer
model = BiLSTM(vocab_size, embedding_dim, hidden_dim, dropout=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.001)  # Add L2 regularization

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state_dict = None

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_data), batch_size):
        batch_X = train_data[i:i + batch_size].squeeze(1)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output.transpose(1, 2), batch_X)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for i in range(0, len(val_data), batch_size):
            batch_X = val_data[i:i + batch_size].squeeze(1)
            output = model(batch_X)
            val_loss += criterion(output.transpose(1, 2), batch_X).item()
        val_loss /= len(val_data)
        val_losses.append(val_loss)

    # Save best model state
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state_dict = model.state_dict()

    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}')

# Load the best model state
model.load_state_dict(best_model_state_dict)

# Plotting training and validation loss
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()
