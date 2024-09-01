import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

class TrainerExecutor:
    """
    Classe responsável por criar e treinar um modelo de rede neural convolucional
    para classificar desenhos de relógios em três categorias: Alzheimer, Parkinson
    e Neurotípico (normal).
    """
    def __init__(self, dataset_path: str) -> None:
        """
        Inicializa o TrainerExecutor com o caminho do diretório de dados.
        """
        self.__dataset_path = dataset_path
        self.__img_size = (200, 200)  # Tamanho das imagens

    def execute(self) -> None:
        """
        Executa o processo de treinamento do modelo, que inclui a preparação dos dados,
        construção do modelo, e treinamento com os dados fornecidos.
        """
        train_data, train_labels, val_data, val_labels = self.__prepare_data()
        self.__model_training(train_data, train_labels, val_data, val_labels)

    def __prepare_data(self):
        """
        Prepara e retorna os dados de treinamento e validação.
        """
        def load_images_from_folder(folder, label):
            images = []
            labels = []
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.__img_size)
                    img_array = np.asarray(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
            return images, labels

        # Carregar imagens de cada classe
        neurotipica_images, neurotipica_labels = load_images_from_folder(os.path.join(self.__dataset_path+ "/train/", 'neurotipica'), 0)
        parkinson_images, parkinson_labels = load_images_from_folder(os.path.join(self.__dataset_path + "/train/", 'parkinson'), 1)
        alzheimer_images, alzheimer_labels = load_images_from_folder(os.path.join(self.__dataset_path + "/train/", 'alzheimer'), 2)

        # Combinar e dividir os dados em treino e validação
        all_images = np.array(neurotipica_images + parkinson_images + alzheimer_images)
        all_labels = np.array(neurotipica_labels + parkinson_labels + alzheimer_labels)

        # Normalização
        all_images = all_images / 255.0

        # Dividir os dados
        train_data, val_data, train_labels, val_labels = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42
        )

        return train_data, train_labels, val_data, val_labels

    def __model_training(self, train_data, train_labels, val_data, val_labels) -> None:
        """
        Constrói e treina um modelo de rede neural convolucional para classificação
        de imagens. O modelo é salvo após o treinamento.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: Alzheimer, Parkinson, Neurotípico
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        def scheduler(epoch, lr):
            if epoch % 10 == 0 and epoch != 0:
                lr = lr * 0.9
            return lr

        lr_scheduler = LearningRateScheduler(scheduler)

        history = model.fit(
            train_data, train_labels,
            epochs=50,
            validation_data=(val_data, val_labels),
            callbacks=[lr_scheduler]
        )
        model.save("trained_t3.h5") 

if __name__ == '__main__':
    dataset_path = "/home/marcus/go/src/github.com/PyMarcus/trabalho2_marcus/dataset"
    te = TrainerExecutor(dataset_path)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU(s) detectada(s): {physical_devices}")
    else:
        print("Nenhuma GPU detectada.")
    
    te.execute()
