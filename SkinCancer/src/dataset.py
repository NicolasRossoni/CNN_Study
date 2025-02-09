import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

# Define as transformações para pré-processamento das imagens
transform = Compose([
    RandomHorizontalFlip(p=0.5),  # Aplica espelhamento horizontal aleatório
    RandomRotation(10),  # Rotação aleatória de até 10 graus
    ColorJitter(brightness=0.2, contrast=0.2),  # Ajusta brilho e contraste aleatoriamente
    Resize((28, 28)),  # Redimensiona para 28x28 antes de converter para tensor
    ToTensor(),  # Converte a imagem para tensor PyTorch
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliza os valores dos pixels
])

# Dicionário de rótulos (caso as classes estejam em formato de string no dataset)
labels_dict = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6
}

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Inicializa o dataset HAM10000.

        Args:
            csv_file (str): Caminho para o arquivo CSV contendo os dados das imagens.
            transform (callable, opcional): Transformação opcional a ser aplicada nas imagens.
        """
        self.data = pd.read_csv(csv_file)  # Carrega os dados do CSV

        # Garante que apenas os pixels e rótulos sejam considerados
        self.image_data = self.data.iloc[:, :-1].values  # Valores dos pixels das imagens
        self.labels = self.data.iloc[:, -1].values  # Rótulos das imagens (já numéricos)

        self.transform = transform  # Define a transformação a ser aplicada

    def __len__(self):
        """ Retorna o número total de amostras no dataset """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retorna uma única amostra do dataset (imagem e rótulo).
        
        Args:
            idx (int): Índice da amostra.
        
        Returns:
            tuple: (imagem processada, rótulo)
        """
        pixels = np.array(self.image_data[idx], dtype=np.uint8).reshape(28, 28, 3)  # Converte os pixels para uma imagem RGB
        label = int(self.labels[idx])  # Converte o rótulo para inteiro

        if self.transform:
            pixels = self.transform(pixels)  # Aplica transformações na imagem

        return pixels, label  # Retorna a imagem e o rótulo correspondente

def get_ham10000_dataloaders(batch_size=64, data_dir="data/HAM10000", split_ratio=0.8):
    """
    Carrega o conjunto de dados HAM10000 e retorna os DataLoaders para treino e teste.

    Args:
        batch_size (int): Tamanho do lote para treinamento e teste.
        data_dir (str): Caminho para a pasta do dataset.
        split_ratio (float): Proporção dos dados a serem utilizados para treino.

    Returns:
        tuple: (train_loader, test_loader) DataLoaders para treinamento e teste.
    """
    transform = Compose([
        ToTensor(),  # Converte a imagem para tensor PyTorch
        Resize((28, 28)),  # Redimensiona para 28x28
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliza os valores dos pixels
    ])

    # Carrega o dataset
    dataset = HAM10000Dataset(f"{data_dir}/hmnist_28_28_RGB.csv", transform=transform)

    # Divide o dataset em conjunto de treino e teste
    train_size = int(split_ratio * len(dataset))  # Define o tamanho do conjunto de treino
    test_size = len(dataset) - train_size  # Define o tamanho do conjunto de teste
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Realiza a divisão

    # Cria os DataLoaders para carregar os dados em lotes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Embaralha os dados de treino
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Não embaralha os dados de teste

    return train_loader, test_loader
