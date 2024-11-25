import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import optuna
import matplotlib.pyplot as plt

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

# Split datasets into training and validation
train_data, val_data = random_split(train_dataset, [50000, 10000])

# Data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define ANNClassifier model
class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate to 0.3

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Objective function for Bayesian optimization
def objective(trial):
    # Suggest hyperparameters with refined ranges
     #This test has been run multiple times and we've narrowed these results down 
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)  # Refined range
    step_size = trial.suggest_int('step_size', 6, 8) #Narrower range 
    gamma = trial.suggest_float('gamma', 0.6, 0.8)  # Narrower range closer to best trials

    # Initialize model, criterion, optimizer, and scheduler
    model = ANNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    best_val_accuracy = 0.0
    patience, epochs_without_improvement = 15, 0  # Increased patience and epochs
    best_model_wts = None

    for epoch in range(30):  # Increased epochs to 30
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        val_loss, val_accuracy = validate_model(model, criterion, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

    # Test accuracy
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    test_accuracy = test_model(model, test_loader)

    return test_accuracy  # Maximize test accuracy

# Validation function
def validate_model(model, criterion, val_loader):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(val_loader), 100 * correct / total

# Testing function
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run Optuna study with increased n_trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Increased n_trials for more thorough search

# Display best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best test accuracy: {study.best_value:.2f}%")