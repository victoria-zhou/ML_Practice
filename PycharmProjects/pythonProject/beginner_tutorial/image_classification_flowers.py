import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import ssl
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def dwl_exp_data():

    print('download_data')

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    # get_file returns Path to the downloaded file
    data_dir = tf.keras.utils.get_file('flower_photos', origin=url, untar=True)
    # pathlib.Path make data_dir path a Path object
    data_dir = pathlib.Path(data_dir)

    # helpful information
    # data_dir.glob('*/*.jpg') returns the files of the data_dir directory/path
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    print('Rose Picture as an example')
    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0])) # can open a string representing path, but not a Path object, thus needs to do str()

    print('Tulip Picture as an example')
    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))  # can open a string representing path, but not a Path object, thus needs to do str()

    return data_dir

def keras_prepro_load(data_dir):
    print('keras_preprocessing_and_load_data')

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    return train_ds, val_ds, class_names, img_height, img_width

def visualize(train_ds, class_names):
    print('visualize')

    plt.figure(figsize=(10, 10))

    for images, labels in train_ds.take(1):
        print('images shape:\n')
        print(images.shape)
        print('labels shape:\n')
        print(labels.shape)

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')

    return

def config_datasets(train_ds, val_ds):
    print('config_datasets')

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def create_model(img_height, img_width):
    print('create_model')

    num_classes = 5
    model = tf.keras.models.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same',activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    return model

def train_model(train_ds, val_ds, model, epochs):
    print('train_model')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history, epochs

def visualize_results(history, epochs):
    print('visualise_result')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()

    return

def run():
    data_dir = dwl_exp_data()
    train_ds, val_ds, class_names, img_height, img_width = keras_prepro_load(data_dir)
    visualize(train_ds, class_names)
    train_ds, val_ds = config_datasets(train_ds, val_ds)
    model = create_model(img_height, img_width)
    history, epochs = train_model(train_ds, val_ds, model, 5)
    visualize_results(history, epochs)
    return

if __name__ == '__main__':
    run()














