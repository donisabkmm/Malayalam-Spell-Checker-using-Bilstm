import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the dataset
words = ['hello', 'world', 'spell', 'check', 'hi', 'donis', 'billan', 'document']
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(set(''.join(words))))}
max_len = max(len(word) for word in words)
X = [[char_to_idx[char] for char in word] for word in words]
X = [[x + [0] * (max_len - len(x))] for x in X]  # Pad sequences
X = torch.tensor(X, dtype=torch.long)

# Define the BiLSTM model with dropout and L2 regularization
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
        output = self.dropout(output)  # Apply dropout
        output = self.fc(output)
        return output

# Define hyperparameters
vocab_size = len(char_to_idx) + 1
embedding_dim = 50
hidden_dim = 64
num_epochs = 10
batch_size = 32

# Split the data into training and validation sets
split = int(0.8 * len(X))
train_data, val_data = X[:split], X[split:]

# Initialize the model, loss function, and optimizer
model = BiLSTM(vocab_size, embedding_dim, hidden_dim, dropout=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.001)  # Add L2 regularization

# Training loop
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_data), batch_size):
        batch_X = train_data[i:i + batch_size].squeeze(1)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output.transpose(1, 2), batch_X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for i in range(0, len(val_data), batch_size):
            batch_X = val_data[i:i + batch_size].squeeze(1)
            output = model(batch_X)
            val_loss += criterion(output.transpose(1, 2), batch_X).item()
        val_loss /= len(val_data)
    train_losses.append(loss.item())
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')

# Save the trained model
torch.save(model.state_dict(), 'spellcheck_model.pth')

# Plot training and validation losses
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Load the trained model
model.load_state_dict(torch.load('spellcheck_model.pth'))

# Define a function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.argmax(axis=2).flatten()
    return (y_true_flat == y_pred_flat).sum() / len(y_true_flat)

# Create DataLoader for validation data
val_dataset = TensorDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model
model.eval()
val_losses = []
all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch_X in val_loader:
        batch_X = batch_X[0]
        batch_X = batch_X.squeeze(1)
        output = model(batch_X)
        loss = criterion(output.transpose(1, 2), batch_X)
        val_losses.append(loss.item())

        # Store predictions and labels for accuracy calculation
        all_val_preds.append(output.detach().numpy())
        all_val_labels.append(batch_X.detach().numpy())

# Combine predictions and labels into numpy arrays
all_val_preds = np.concatenate(all_val_preds)
all_val_labels = np.concatenate(all_val_labels)

# Calculate validation accuracy
val_accuracy = calculate_accuracy(all_val_labels, all_val_preds)
print(f'Validation Accuracy: {val_accuracy}')

# Compute mean validation loss
mean_val_loss = np.mean(val_losses)
print(f'Validation Loss: {mean_val_loss}')

# Analyze fit
if mean_val_loss < train_losses[-1]:
    print("Model may be underfitting.")
elif mean_val_loss > train_losses[-1]:
    print("Model may be overfitting.")
else:
    print("Model is fitting well.")
