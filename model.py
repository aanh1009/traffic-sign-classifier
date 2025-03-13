import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os

# Data transformation
def get_data_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load dataset
def load_dataset(data_transforms, portion=0.1, batch_size=64):
    train_dataset = GTSRB(root='data', split='train', download=True, transform=data_transforms)

    total_size = len(train_dataset)
    subset_size = int(portion * total_size)
    indices = np.random.choice(total_size, subset_size, replace=False)

    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Define model
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, padding=1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, padding=1, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, padding=1, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
def train_model(model, train_loader, epochs=24, lr=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Evaluate the model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on test images: {accuracy:.2f}%')
    return accuracy

# Save model
def save_model(model, filename="model.pt"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Load model
def load_model(filename="model.pt"):
    model = TrafficSignNet()
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model loaded from {filename}")
    return model

# Save transformations
def save_transforms(transform, filename="saved_transforms.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump({"data_transforms": transform}, file)
    print(f"Transformations saved to {filename}")

# Load transformations
def load_transforms(filename="saved_transforms.pkl"):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Transformations loaded from {filename}")
    return data["data_transforms"]

# Convert model to TorchScript
def export_torchscript_model(model, filename="model_scripted.pt"):
    model_scripted = torch.jit.script(model)
    model_scripted.save(filename)
    print(f"Model exported to TorchScript format: {filename}")

# Main function
def main():
    # Load data
    data_transforms = get_data_transforms()
    train_loader, val_loader = load_dataset(data_transforms)

    # Initialize model
    model = TrafficSignNet()

    # Train model
    train_model(model, train_loader)

    # Evaluate model
    evaluate_model(model, val_loader)

    # Save model and transformations
    save_model(model)
    save_transforms(data_transforms)

    # Export TorchScript model
    export_torchscript_model(model)

if __name__ == "__main__":
    main()
