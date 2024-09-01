import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

class TrainerExecutor:
    """
    Classe responsável por criar e treinar um modelo de rede neural convolucional
    para classificar desenhos de relógios em três categorias: Alzheimer, Parkinson
    e Neurotípico (normal).
    """
    def __init__(self, dataset_path: str, validation_path: str) -> None:
        """
        Inicializa o TrainerExecutor com os caminhos dos diretórios de dados.
        """
        self.__dataset_path = dataset_path
        self.__validation_path = validation_path

    def execute(self) -> None:
        """
        Executa o processo de treinamento do modelo, que inclui a preparação dos dados,
        construção do modelo, e treinamento com os dados fornecidos.
        """
        self.__model_training()

    def __prepare_data(self) -> ImageDataGenerator:
        """
        Prepara e retorna um gerador de dados de treinamento com aumento de dados.
        """
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

    def __generators(self) -> tuple:
        """
        Cria e retorna geradores de dados para treinamento e validação.
        """
        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = self.__prepare_data().flow_from_directory(
            self.__dataset_path,
            target_size=(150, 150),
            batch_size=32,
            color_mode='grayscale',  
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.__validation_path,
            target_size=(150, 150),
            batch_size=32,
            color_mode='grayscale', 
            class_mode='categorical'
        )
        return train_generator, validation_generator
    
    def __model_training(self) -> None:
        """
        Constrói e treina um modelo de rede neural convolucional para classificação
        de imagens. O modelo é salvo após o treinamento.
        """
        model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
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
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        def scheduler(epoch, lr):
            if epoch % 10 == 0 and epoch != 0:
                lr = lr * 0.9
            return lr

        lr_scheduler = LearningRateScheduler(scheduler)
    
        train_gen, validation_gen = self.__generators()
    
        history = model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            epochs=1500,  
            validation_data=validation_gen,
            validation_steps=validation_gen.samples // validation_gen.batch_size,
            callbacks=[lr_scheduler]
        )
        model.save("trained_t2.h5") 

if __name__ == '__main__':
    import os
    
    dataset_path = "/home/marcus/go/src/github.com/PyMarcus/trabalho2_marcus/dataset/train"
    validation_path = "/home/marcus/go/src/github.com/PyMarcus/trabalho2_marcus/dataset/validation"
    te = TrainerExecutor(dataset_path, validation_path)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU(s) detectada(s): {physical_devices}")
    else:
        print("Nenhuma GPU detectada.")
    
    te.execute()
