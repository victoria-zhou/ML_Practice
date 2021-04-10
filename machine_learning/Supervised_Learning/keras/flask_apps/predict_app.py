#!/usr/bin/env python
# coding: utf-8

# In[1]:


import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


def get_model():
    # global model
    model = load_model('VGG16_cats_and_dogs.h5')
    model._make_predict_function()
    print(" * Model loaded!")
    return model

# In[5]:


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


# In[6]:


print(" * Loading Keras model...")
model = get_model()
# graph = tf.get_default_graph()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image'].split(',')[1]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    import ipdb; ipdb.set_trace()
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    return jsonify(response)
