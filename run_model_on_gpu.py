
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam


def load_train(path='/datasets/faces/final_files/'):
    
    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """
    
    # coloca tu código aquí
    train_datagen = ImageDataGenerator(validation_split=0.25,rescale=1/255,fill_mode='nearest')
    train_gen_flow = train_datagen.flow_from_dataframe(dataframe=pd.read_csv('/datasets/faces/labels.csv'),
                                                        directory='/datasets/faces/final_files/',
                                                        x_col='file_name',
                                                        y_col='real_age',
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='raw',
                                                        subset='training',
                                                        seed=12345)

    return train_gen_flow


def load_test(path='/datasets/faces/final_files/'):
    
    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta
    """
    
    #  coloca tu código aquí
    test_datagen = ImageDataGenerator(validation_split=0.25,rescale=1/255,fill_mode='nearest')
    test_gen_flow = test_datagen.flow_from_dataframe(dataframe=pd.read_csv('/datasets/faces/labels.csv'),
                                                        directory='/datasets/faces/final_files/',
                                                        x_col='file_name',
                                                        y_col='real_age',
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='raw',
                                                        subset='validation',
                                                        seed=12345)

    return test_gen_flow


def create_model(input_shape):
    
    """
    Define el modelo
    """
    
    #  coloca tu código aquí
    backbone = ResNet50(input_shape=input_shape, weights='imagenet',include_top=False)
    
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Entrena el modelo dados los parámetros
    """
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    return model


