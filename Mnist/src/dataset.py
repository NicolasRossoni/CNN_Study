# Carrega e pré-processa o conjunto de dados MNIST

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64, data_dir="data"):
    """
    Faz o download do conjunto de dados MNIST e retorna os DataLoaders para treino e teste.

    Args:
        batch_size (int): O tamanho do lote para carregar os dados.
        data_dir (str): Diretório onde os dados do MNIST serão armazenados.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader): DataLoaders para treino e teste.
    """

    # Define transformações (converte imagens para tensores e normaliza)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte as imagens para tensores PyTorch
        transforms.Normalize((0.1307,), (0.3081,))  # Normaliza usando a média e o desvio padrão do MNIST
    ])

    # Faz o download e carrega o conjunto de dados de treinamento
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True
    )

    # Faz o download e carrega o conjunto de dados de teste
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True
    )

    # Cria os DataLoaders para carregar os dados em lotes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Embaralha os dados de treino
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Não embaralha os dados de teste

    return train_loader, test_loader

# Testa o carregamento do conjunto de dados
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()
    print(f"MNIST carregado: {len(train_loader.dataset)} amostras de treino, {len(test_loader.dataset)} amostras de teste")
