# Define a arquitetura do modelo CNN para MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # Camadas convolucionais
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Primeira camada convolucional
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Segunda camada convolucional
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Camada de Max Pooling para redução de dimensionalidade

        # Camadas totalmente conectadas (Fully Connected)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Primeira camada totalmente conectada
        self.fc2 = nn.Linear(128, 10)  # Camada de saída com 10 classes (dígitos de 0 a 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Aplicação da primeira convolução, ReLU e MaxPooling
        x = self.pool(F.relu(self.conv2(x)))  # Aplicação da segunda convolução, ReLU e MaxPooling
        x = x.view(-1, 64 * 7 * 7)  # Achatamento dos mapas de características para entrada na camada totalmente conectada
        x = F.relu(self.fc1(x))  # Aplicação da primeira camada totalmente conectada com ativação ReLU
        x = self.fc2(x)  # Camada de saída (logits)
        return x

# Testa a arquitetura do modelo
if __name__ == "__main__":
    model = MNIST_CNN()
    print(model)
