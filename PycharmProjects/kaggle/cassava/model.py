import tensorflow as tf
import os
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():

    train_datagen = ImageDataGenerator(rescale=1.0/255)

    train_dir = './data/train'
    train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(300, 300),
        batch_size=30,
        class_mode='categorical'
    )

    return train_generator

def build_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(300, 300, 3), kernel_regularizer='l2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(train_generator, model):

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=10,
        verbose=1
    )

    return history

def visualise(history, epochs=10):

    history = history.history
    epoch = [i+1 for i in range(1, 11)]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    loss = history['loss']
    accuracy = history['accuracy']
    axes[0].plot(epoch, loss, label='loss')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Epoch')

    axes[1].plot(epoch, accuracy, label='accuracy')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Epoch')


    return

if __name__ == '__main__':
    train_generator = load_data()
    model = build_model()
    history = train_model(train_generator, model)
    visualise(history)
