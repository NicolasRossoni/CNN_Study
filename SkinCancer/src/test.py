import torch
from model import HAM_CNN
from dataset import get_ham10000_dataloaders

# Load the model for evaluation
def load_model(model_path, device):
    model = HAM_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Evaluate the model on the test dataset
def evaluate_model(model, test_loader, device):
    total = 0
    correct = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = outputs.max(1)

            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test dataset: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    model_path = "models/ham_cnn.pth"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset (get both train and test loaders)
    train_loader, test_loader = get_ham10000_dataloaders(batch_size=batch_size)

    # Load the trained model
    model = load_model(model_path, device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)
