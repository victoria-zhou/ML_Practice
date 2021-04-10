import os
import io
import shutil
import string
import tensorflow as tf
import ssl

import tensorboard

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def download_dataset():

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz",
               url,
               untar=True)

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    os.listdir(dataset_dir)

    # check train dir
    train_dir = os.path.join(dataset_dir, 'train')
    os.listdir(train_dir)

    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    return train_dir

def process_dataset(train_dir):

    batch_size = 1024
    seed = 123

    train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        seed=seed,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        seed=seed,
        validation_split=0.2,
        subset='validation'
    )

    for text_batch, label_batch in train_ds.take(1):
        for i in range(5):
            print(label_batch[i].numpy(), text_batch.numpy()[i])

    return train_ds, val_ds

def config_dataset(train_ds, val_ds):

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def get_emd_layer():

    embedding_layer = tf.keras.layers.Embedding(1000, 5)

    return embedding_layer

def custom_standardization(input_data):

    import re
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

    return tf.strings.regex_replace(stripped_html,
                                      '[%s]' % re.escape(string.punctuation), '')

def preprocessing(train_ds):
    vocab_size = 10000
    sequence_length = 100

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    return vectorize_layer, vocab_size

def get_model(vectorize_layer, vocab_size):

    embedding_dim = 16
    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name='embedding'),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def train_model(model, train_ds, val_ds):

    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model.fit(train_ds,
    #           validation_data = val_ds,
    #           epochs = 15,
    #           callbacks = [tensorboard_callback])

    model.fit(train_ds,
              validation_data = val_ds,
              epochs = 15)

    model.summary()

    return

def retrive_and_save(model, vectorize_layer):

    weights = model.get_layer('embedding').get_weights()[0]
    # The get_vocabulary() function provides the vocabulary to build a metadata file with one token per line.
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('../data/word_embeddings/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('../data/word_embeddings/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0: continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()

    return

# def vis_tensorboard():
#     % load_ext tensorboard
#     % tensorboard --logdir logs
#
#     from tensorboard import program
#     tb = program.TensorBoard()
#     tb.configure(argv=[None, '--logdir', logs])
#     url = tb.launch()


def run():
    train_dir = download_dataset()
    train_ds, val_ds = process_dataset(train_dir)
    train_ds, val_ds = config_dataset(train_ds, val_ds)

    embedding_layer = get_emd_layer()
    vectorize_layer, vocab_size = preprocessing(train_ds)
    model = get_model(vectorize_layer, vocab_size)

    train_model(model, train_ds, val_ds)
    retrive_and_save(model, vectorize_layer)

    return


if __name__ == '__main__':
        run()

