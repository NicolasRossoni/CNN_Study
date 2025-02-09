import torch
import torch.nn as nn
import torch.nn.functional as F

class HAM_CNN(nn.Module):
    def __init__(self):
        super(HAM_CNN, self).__init__()

        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Entrada com 3 canais (RGB), saída com 32 filtros
        self.bn1 = nn.BatchNorm2d(32)  # Normalização em lote para estabilizar o treinamento

        # Segunda camada convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Entrada com 32 canais, saída com 64 filtros
        self.bn2 = nn.BatchNorm2d(64)

        # Camada de MaxPooling para reduzir a dimensão espacial
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduz a dimensão pela metade

        # Terceira camada convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Entrada com 64 canais, saída com 128 filtros
        self.bn3 = nn.BatchNorm2d(128)

        # Quarta camada convolucional
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Entrada com 128 canais, saída com 256 filtros
        self.bn4 = nn.BatchNorm2d(256)

        # **Correção: Calcula automaticamente o tamanho correto da entrada para a camada totalmente conectada**
        self._to_linear = None
        self._get_conv_output_size()  # Calcula dinamicamente o tamanho da saída das convoluções

        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(self._to_linear, 512)  # Primeira camada totalmente conectada
        self.fc2 = nn.Linear(512, 7)  # Camada de saída com 7 classes (lesões da HAM10000)

    def _get_conv_output_size(self):
        """ Calcula dinamicamente o tamanho da saída da última camada convolucional """
        x = torch.randn(1, 3, 28, 28)  # Entrada fictícia com tamanho 28x28 e 3 canais (RGB)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Passa pela primeira camada convolucional e MaxPooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Segunda camada convolucional e MaxPooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Terceira camada convolucional e MaxPooling
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Quarta camada convolucional e MaxPooling
        self._to_linear = x.numel()  # Obtém o número total de elementos no tensor achatado

    def forward(self, x):
        """ Define a passagem para frente da rede """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Primeira convolução + ReLU + BatchNorm + MaxPooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Segunda convolução + ReLU + BatchNorm + MaxPooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Terceira convolução + ReLU + BatchNorm + MaxPooling
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Quarta convolução + ReLU + BatchNorm + MaxPooling
        x = x.view(-1, self._to_linear)  # **Correção: Achata dinamicamente a saída da convolução**
        x = F.relu(self.fc1(x))  # Primeira camada totalmente conectada com ReLU
        x = self.fc2(x)  # Camada de saída (logits para 7 classes)
        return x

# Teste da arquitetura do modelo
if __name__ == "__main__":
    model = HAM_CNN()
    print(model)
