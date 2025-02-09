# Testa o modelo com um canvas interativo

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from model import MNIST_CNN
import torchvision.transforms as transforms

# Carrega o modelo treinado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define o dispositivo (GPU ou CPU)
model = MNIST_CNN()  # Inicializa o modelo
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device, weights_only=True))  # Carrega os pesos do modelo
model.to(device)  # Move o modelo para o dispositivo
model.eval()  # Define o modelo para modo de avaliação

# Transformações para pré-processamento da imagem desenhada
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Converte para escala de cinza
    transforms.Resize((28, 28)),  # Redimensiona para 28x28 pixels (tamanho do MNIST)
    transforms.ToTensor(),  # Converte para tensor PyTorch
    transforms.Normalize((0.1307,), (0.3081,))  # Normaliza usando a média e desvio padrão do MNIST
])

# Função para fazer a predição do dígito desenhado
def predict_digit(image):
    image_tensor = transform(image).unsqueeze(0).to(device)  # Aplica transformações e adiciona dimensão batch
    output = model(image_tensor)  # Passa a imagem pelo modelo
    _, predicted = output.max(1)  # Obtém a classe prevista
    return predicted.item()

# Interface gráfica com Tkinter
class DigitCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predição de Dígitos MNIST")

        # Canvas para desenhar
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self.draw)  # Detecta movimento do mouse ao pressionar botão esquerdo

        # Botão para fazer a predição
        self.predict_button = tk.Button(self.root, text="Prever", command=self.make_prediction)
        self.predict_button.grid(row=1, column=0, pady=10)

        # Rótulo para exibir a predição
        self.prediction_label = tk.Label(self.root, text="Desenhe um dígito e clique em Prever!", font=("Helvetica", 16))
        self.prediction_label.grid(row=2, column=0, pady=10)

        # Botão para limpar o canvas
        self.clear_button = tk.Button(self.root, text="Limpar", command=self.clear_canvas)
        self.clear_button.grid(row=3, column=0, pady=10)

        # Imagem PIL para armazenar o desenho
        self.image = Image.new("L", (280, 280), "white")  # Imagem em escala de cinza (L)
        self.draw_instance = ImageDraw.Draw(self.image)  # Objeto para desenhar na imagem

    def draw(self, event):
        """ Função chamada ao desenhar no canvas """
        x, y = event.x, event.y
        radius = 8  # Tamanho do traço
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black")  # Desenha no canvas
        self.draw_instance.ellipse([x - radius, y - radius, x + radius, y + radius], fill="black")  # Desenha na imagem PIL

    def make_prediction(self):
        """ Converte o desenho e faz a predição """
        resized_image = self.image.resize((28, 28))  # Redimensiona para 28x28 pixels
        inverted_image = ImageOps.invert(resized_image)  # Inverte as cores (fundo branco, dígito preto)
        prediction = predict_digit(inverted_image)  # Faz a predição com o modelo
        self.prediction_label.config(text=f"Dígito previsto: {prediction}")  # Exibe o resultado na interface

    def clear_canvas(self):
        """ Limpa o canvas e reseta a imagem desenhada """
        self.canvas.delete("all")  # Apaga tudo do canvas
        self.image = Image.new("L", (280, 280), "white")  # Reseta a imagem PIL
        self.draw_instance = ImageDraw.Draw(self.image)  # Cria novo objeto de desenho

# Executa a aplicação Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitCanvasApp(root)
    root.mainloop()
