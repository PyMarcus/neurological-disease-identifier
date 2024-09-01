import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

class ImageClassifier:
    def __init__(self, model_path: str, img_size: tuple = (200, 200)) -> None:
        """
        Inicializa o ImageClassifier com o caminho do modelo e o tamanho da imagem.
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = self.__load_model()
        if self.model is not None:
            print(f"Modelo carregado com sucesso de: {self.model_path}")
        else:
            print(f"Falha ao carregar o modelo.")

    def __load_model(self) -> tf.keras.Model:
        """
        Carrega o modelo treinado a partir do caminho fornecido.
        """
        try:
            model = load_model(self.model_path)
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return None

    def __prepare_image(self, img_path: str) -> np.ndarray:
        """
        Prepara a imagem para a classificação, incluindo redimensionamento e normalização.
        """
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Converter a imagem para RGB se não estiver
            img = img.resize(self.img_size)  # Redimensionar para o tamanho esperado pelo modelo
            
            img_array = np.asarray(img)  # Converter para array NumPy
            img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão de batch
            img_array = img_array / 255.0  # Normalizar a imagem para o intervalo [0, 1]
            
            return img_array
        except Exception as e:
            print(f"Erro ao preparar a imagem: {e}")
            raise

    def classify(self, img_path: str) -> str:
        """
        Classifica a imagem e retorna o nome da classe.
        """
        if self.model is None:
            print("Modelo não carregado.")
            return "Não identificado!"
        
        try:
            img_array = self.__prepare_image(img_path)
            predictions = self.model.predict(img_array)
            print(f"PREDICTIONS: {predictions}")  # Verifique os valores aqui
            class_idx = np.argmax(predictions, axis=1)[0]
            class_names = ['neurotipica', 'parkinson', 'alzheimer']
            if class_idx < len(class_names):
                return class_names[class_idx]
            else:
                print(f"Índice {class_idx} fora dos limites para class_names.")
                return "Não identificado!"
        except Exception as e:
            print(f"Erro durante a classificação: {e}")
            return "Não identificado!"
