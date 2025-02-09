# CNN - Reconhecimento de Dígitos e Diagnóstico de Câncer de Pele

Este repositório contém dois projetos que implementam redes neurais convolucionais (CNNs) para tarefas distintas:
1. Reconhecimento de dígitos do dataset MNIST.
2. Diagnóstico de lesões de pele utilizando o dataset HAM10000.

Ambos os projetos foram desenvolvidos para demonstrar o uso de redes neurais em tarefas de classificação.

---

## Estrutura do Projeto

O projeto está organizado em duas pastas principais, cada uma correspondente a um conjunto de dados:

```
CNN_Study/
├── Mnist/
│   ├── data/                  # Diretório onde o dataset MNIST será armazenado.
│   ├── models/                # Diretório para salvar os modelos treinados.
│   ├── src/                   # Código-fonte para o projeto MNIST.
│       ├── dataset.py         # Manipulação e download do dataset MNIST.
│       ├── interactive_canvas.py  # Interface interativa para desenhar dígitos e testá-los.
│       ├── model.py           # Definição do modelo CNN para MNIST.
│       ├── test.py            # Avaliação do modelo no conjunto de teste.
│       └── train.py           # Treinamento do modelo CNN no dataset MNIST.
├── SkinCancer/
│   ├── data/                  # Diretório onde o dataset HAM10000 será armazenado.
│   ├── models/                # Diretório para salvar os modelos treinados.
│   ├── src/                   # Código-fonte para o projeto SkinCancer.
│       ├── dataset.py         # Manipulação do dataset HAM10000.
│       ├── model.py           # Definição do modelo CNN para diagnóstico de lesões de pele.
│       ├── test.py            # Avaliação do modelo no conjunto de teste.
│       └── train.py           # Treinamento do modelo CNN no dataset HAM10000.
└── README.md                  # Este arquivo explicativo.
```

---

## Descrição dos Arquivos

### Diretório `Mnist/src/`
- **`dataset.py`**: Contém funções para baixar e carregar o dataset MNIST.
- **`interactive_canvas.py`**: Implementa uma interface gráfica para desenhar dígitos e testar o modelo treinado.
- **`model.py`**: Define a arquitetura da rede neural convolucional para classificação dos dígitos.
- **`test.py`**: Avalia o modelo treinado no conjunto de teste e exibe a acurácia.
- **`train.py`**: Realiza o treinamento do modelo no dataset MNIST e salva os pesos.

### Diretório `SkinCancer/src/`
- **`dataset.py`**: Prepara o dataset HAM10000 para treinamento e teste.
- **`model.py`**: Define a arquitetura da rede neural convolucional para classificação de lesões de pele.
- **`test.py`**: Avalia o modelo treinado no conjunto de teste e exibe a acurácia.
- **`train.py`**: Realiza o treinamento do modelo no dataset HAM10000 e salva os pesos.

---

## Dependências

Certifique-se de ter o Python 3.8 ou superior instalado. As dependências podem ser instaladas com o comando abaixo:

```bash
pip install -r requirements.txt
```

Crie um arquivo `requirements.txt` com as seguintes bibliotecas:

```
torch
torchvision
pandas
numpy
Pillow
tkinter
```

---

## Como Rodar os Arquivos

### Projeto MNIST

Antes entre no diretorio Mnist

0. **Baixar a Base de Dados:**

   ```bash
   python3 src/database.py
   ```
1. **Treinar o modelo:**
   ```bash
   python3 src/train.py
   ```
2. **Testar o modelo no conjunto de teste:**
   ```bash
   python3 src/test.py
   ```
3. **Usar a interface interativa para desenhar e testar dígitos:**
   ```bash
   python3 src/interactive_canvas.py
   ```

### Projeto SkinCancer

Antes entre no diretorio SkinCancer

0. **Baixar a Base de Dados:**

   Insira ela manualmente na pasta "SkinCancer/data", após baixa-lá nesse site:
   
   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data

   Os arquivos virão em uma pasta chamada "arquive", tire todos eles dessa pasta e os coloque em "SkinCancer/data".

1. **Treinar o modelo:**
   ```bash
   python3 src/train.py
   ```
2. **Testar o modelo no conjunto de teste:**
   ```bash
   python3 src/test.py
   ```

---

## Observações
- Certifique-se de que os datasets MNIST e HAM10000 estejam armazenados corretamente nos diretórios especificados.
- Os modelos treinados serão salvos na pasta `models/` correspondente a cada projeto.
- A interface interativa do MNIST utiliza `tkinter` para desenhar dígitos e testá-los em tempo real.


