import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

class ImageClassifier:
    def __init__(self, model_path: str, img_size: tuple = (150, 150)) -> None:
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        try:
            self.model = self.__load_model()
        except Exception as e:
            print(f"Error loading model: {e}")

    def __load_model(self) -> tf.keras.Model:
        return load_model(self.model_path)

    def __prepare_image(self, img_path: str) -> np.ndarray:
        img = image.load_img(img_path, color_mode='grayscale', target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalização: Escala de 0 a 1
        img_array /= 255.0
        

        # mean = np.mean(img_array)
        # std = np.std(img_array)
        # img_array = (img_array - mean) / std
        
        return img_array

    def classify(self, img_path: str) -> str:
        try:
            img_array = self.__prepare_image(img_path)
            predictions = self.model.predict(img_array)
            class_idx = np.argmax(predictions, axis=1)[0]
            class_names = ['neurotipica', 'parkinson', 'alzheimer']
            return class_names[class_idx]
        except Exception as e:
            print(f"Error during classification: {e}")
            return "Não identificado!"
