import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_ham10000_dataloaders
from model import HAM_CNN

# Hiperparâmetros
learning_rate = 0.0001  # Taxa de aprendizado reduzida para maior estabilidade
batch_size = 128  # Aumento do tamanho do lote
epochs = 20  # Mais épocas para melhor aprendizado

# Verifica se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carrega os dados de treinamento e teste
train_loader, test_loader = get_ham10000_dataloaders(batch_size=batch_size)

# Inicializa o modelo, a função de perda e o otimizador
model = HAM_CNN().to(device)  # Move o modelo para GPU (se disponível)
criterion = nn.CrossEntropyLoss()  # Função de perda para classificação multiclasse
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Adam com decaimento de peso
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Redução da taxa de aprendizado a cada 5 épocas

# Loop de treinamento
def train():
    model.train()  # Define o modelo para modo de treinamento
    for epoch in range(epochs):
        total_loss = 0  # Acumulador de perda
        correct = 0  # Contador de previsões corretas
        total = 0  # Contador total de amostras

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move os dados para a GPU (se disponível)

            # Passagem para frente (Forward)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Passagem para trás (Backward)
            optimizer.zero_grad()  # Zera os gradientes
            loss.backward()  # Calcula os gradientes
            optimizer.step()  # Atualiza os pesos

            # Atualiza as estatísticas de treinamento
            total_loss += loss.item()
            _, predicted = outputs.max(1)  # Obtém as previsões mais prováveis
            correct += (predicted == labels).sum().item()  # Conta previsões corretas
            total += labels.size(0)  # Atualiza o total de amostras processadas

        # Ajusta a taxa de aprendizado conforme definido no scheduler
        scheduler.step()

        # Calcula a acurácia e imprime os resultados da época
        accuracy = 100 * correct / total
        print(f"Época [{epoch+1}/{epochs}], Perda: {total_loss/len(train_loader):.4f}, Acurácia: {accuracy:.2f}%")

    # Salva o modelo treinado
    torch.save(model.state_dict(), "models/ham_cnn.pth")
    print("Modelo salvo em models/ham_cnn.pth")

# Executa o treinamento se o script for chamado diretamente
if __name__ == "__main__":
    train()
