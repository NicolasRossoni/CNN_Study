# Avalia o modelo treinado

import torch
from model import MNIST_CNN
from dataset import get_mnist_dataloaders

# Carrega o modelo para avaliação
def load_model(model_path, device):
    model = MNIST_CNN()  # Inicializa o modelo
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Carrega os pesos do modelo treinado
    model.to(device)  # Move o modelo para a GPU (se disponível)
    model.eval()  # Define o modelo para modo de avaliação (desativa dropout e batch norm)
    return model

# Avalia o modelo no conjunto de teste
def evaluate_model(model, test_loader, device):
    total = 0  # Contador do total de amostras
    correct = 0  # Contador de previsões corretas

    with torch.no_grad():  # Desativa o cálculo de gradientes para avaliação (economiza memória)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move os dados para a GPU (se disponível)

            # Passagem para frente (Forward)
            outputs = model(images)
            _, predicted = outputs.max(1)  # Obtém a classe prevista para cada amostra

            # Atualiza métricas de avaliação
            total += labels.size(0)  # Atualiza o número total de amostras
            correct += (predicted == labels).sum().item()  # Conta previsões corretas

    # Calcula e exibe a acurácia final no conjunto de teste
    accuracy = 100 * correct / total
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")
    return accuracy

# Execução principal para avaliação do modelo
if __name__ == "__main__":
    # Hiperparâmetros
    batch_size = 64
    model_path = "models/mnist_cnn.pth"

    # Configuração do dispositivo (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carrega o conjunto de dados de teste
    _, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    # Carrega o modelo treinado
    model = load_model(model_path, device)

    # Avalia o modelo no conjunto de teste
    evaluate_model(model, test_loader, device)
