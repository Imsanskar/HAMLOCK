import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from generate_poisoned_model import BaselineMNISTNetwork
from generate_poisoned_model import PoisonedMNIST

# Load the clean MNIST test dataset
def load_clean_mnist_test(batch_size=64):
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to test the model
def test_model(model, test_loader, description=""):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    loss_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = loss_total / len(test_loader)
    print(f"{description} Accuracy: {accuracy:.2f}%")
    print(f"{description} Average Test Loss: {avg_loss:.4f}")

    return accuracy

# Load the trained poisoned model
model = BaselineMNISTNetwork()
model.load_state_dict(torch.load("mnist_weak_trigger.pth", map_location=torch.device("cpu")))

# Load clean test dataset
clean_test_loader = load_clean_mnist_test()

# Load poisoned test dataset using existing class
poisoned_test_dataset = PoisonedMNIST(train=False, poison_fraction=1.0)  # Ensure test dataset
poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False)

# Evaluate on clean test dataset
test_model(model, clean_test_loader, description="Clean Test")

# Evaluate on poisoned test dataset
test_model(model, poisoned_test_loader, description="Poisoned Test")
