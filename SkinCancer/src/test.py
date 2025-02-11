import torch
import random
import numpy as np
from model import HAM_CNN
from dataset import get_ham10000_dataloaders

# Fixar semente para resultados reproduzíveis
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Carregar o modelo para avaliação
def load_model(model_path, device):
    model = HAM_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Avaliar o modelo no conjunto de teste
def evaluate_model(model, test_loader, device):
    total = 0
    correct = 0

    with torch.no_grad():  # Desabilitar cálculo de gradientes para avaliação
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Passagem para frente
            outputs = model(images)
            _, predicted = outputs.max(1)

            # Atualizar métricas
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Hiperparâmetros
    batch_size = 64
    model_path = "models/ham_cnn.pth"

    # Configuração do dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carregar o conjunto de dados
    train_loader, test_loader = get_ham10000_dataloaders(batch_size=batch_size)

    # Verificar tamanho do conjunto de teste
    print(f"Número de exemplos no conjunto de teste: {len(test_loader.dataset)}")

    # Carregar o modelo treinado
    model = load_model(model_path, device)

    # Avaliar o modelo
    evaluate_model(model, test_loader, device)