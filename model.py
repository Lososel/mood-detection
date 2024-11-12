import os
import torch
import torchaudio
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Path to RAVDESS dataset
dataset_path = '/Users/elmira/Desktop/AML/Audio_Song_Actors_01-24'

# Define emotion mapping based on RAVDESS labels
emotion_mapping = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Function to extract features from an audio file
def extract_features(audio_path, max_length=200):
    waveform, sample_rate = torchaudio.load(audio_path)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc_features = mfcc_transform(waveform)
    
    # Pad or truncate to max_length
    mfcc_features = mfcc_features[:, :, :max_length] if mfcc_features.shape[2] >= max_length else torch.nn.functional.pad(mfcc_features, (0, max_length - mfcc_features.shape[2]))
    
    return mfcc_features.mean(dim=0).flatten().numpy()  # Average across channels and flatten

# Load dataset and prepare features and labels
features_list = []
labels = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            emotion_code = file.split("-")[2]  # Extract emotion code
            emotion_label = emotion_mapping[emotion_code]
            
            # Extract MFCC features
            mfcc_features = extract_features(file_path)
            features_list.append(mfcc_features)
            labels.append(emotion_code)  # Using code for numeric encoding

# Pad features to have the same length
max_length = max(feature.shape[0] for feature in features_list) // 13
padded_features = [
    np.pad(feature, (0, max_length * 13 - len(feature)), mode='constant')
    if len(feature) < max_length * 13 else feature
    for feature in features_list
]

# Convert to numpy arrays and encode labels as integers
features = np.array(padded_features, dtype=np.float32)
labels = np.array([int(label) - 1 for label in labels])  # Make labels 0-indexed

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple neural network for mood detection
class MoodClassifier(nn.Module):
    def __init__(self, input_size):
        super(MoodClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Increased neurons
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, len(emotion_mapping))  # Output layer for 8 emotions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Added dropout
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, loss function, and optimizer
input_size = max_length * 13
model = MoodClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Added learning rate scheduler

# Training loop
num_epochs = 30  # Increased epochs
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    scheduler.step()  # Step the scheduler

# Test the model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Function to detect mood of a song
def detect_mood(audio_path):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    features = extract_features(audio_path, max_length=max_length)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output.data, 1)
        emotion = list(emotion_mapping.values())[predicted.item()]
    print(f"Detected mood: {emotion}")

# Example usage
detect_mood('/Users/elmira/Desktop/AML/audio/smallville-music_radiohead-creep.mp3')
