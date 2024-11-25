import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

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

# Define the models
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class CentralCNN(nn.Module):
    def __init__(self):
        super(CentralCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)  # Corrected dimensions
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_and_evaluate(ModelClass, train_loader, val_loader, test_loader, epochs=20, learning_rate=0.001, step_size=5, gamma=0.1, patience=5):
    model = ModelClass()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0

    # Display the training parameters
    print(f"\nTraining {ModelClass.__name__} with learning rate: {learning_rate}, step size: {step_size}, gamma: {gamma}")

    for epoch in range(epochs):
        model.train()
        running_train_loss, train_correct, train_total = 0.0, 0, 0

        # Training loop with progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        val_loss, val_accuracy = validate_model(model, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print training and validation metrics along with learning parameters
        print(f"Epoch [{epoch+1}/{epochs}] - lr: {learning_rate}, step size: {step_size}, gamma: {gamma}")
        print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"    Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    test_accuracy = test_model(model)
    return train_losses, val_losses, val_accuracies, test_accuracy


def validate_model(model, criterion):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

def test_model(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Plotting function
def plot_comparison(metrics):
    # Combined Validation Accuracy and Loss Plots
    plt.figure(figsize=(14, 6))
    
    # Validation Accuracy Comparison
    plt.subplot(2, 1, 1)
    for metric in metrics:
        plt.plot(metric['val_accuracies'], label=f"{metric['name']} Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy Comparison')
    
    # Loss Comparison
    plt.subplot(2, 1, 2)
    for metric in metrics:
        plt.plot(metric['train_losses'], label=f"{metric['name']} Training Loss")
        plt.plot(metric['val_losses'], label=f"{metric['name']} Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Comparison')
    
    plt.tight_layout()
    plt.show()

    # Individual Plots for Each Model
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Individual Validation Accuracy Plot
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(metric['val_accuracies']) + 1), metric['val_accuracies'], label="Validation Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f"{metric['name']} - Validation Accuracy")
        plt.legend()
        
        # Individual Loss Plot (Training and Validation)
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(metric['train_losses']) + 1), metric['train_losses'], label="Training Loss")
        plt.plot(range(1, len(metric['val_losses']) + 1), metric['val_losses'], label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"{metric['name']} - Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Initialize device, define models with individual hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [
    ("SimpleNN", SimpleNN, 0.0005, 6, 0.1),  # (model name, model class, learning rate, step size, gamma)
    ("CentralCNN", CentralCNN, 0.001, 5, 0.25),
    ("ANNClassifier", ANNClassifier, 0.0009, 7, 0.6)
]
# Train and evaluate each model with individual learning rate, step size, and gamma
metrics = []
for name, model_class, lr, step_size, gamma in models:
    print(f"\nTraining {name} with learning rate: {lr}, step size: {step_size}, gamma: {gamma}")
    train_losses, val_losses, val_accuracies, test_accuracy = train_and_evaluate(
        model_class, train_loader, val_loader, test_loader,
        epochs=20, learning_rate=lr, step_size=step_size, gamma=gamma
    )

    metrics.append({
        "name": name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy
    })
    print(f"{name} Test Accuracy: {test_accuracy:.2f}%")

# Plot comparisons
plot_comparison(metrics)