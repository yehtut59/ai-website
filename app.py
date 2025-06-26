


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
import os
from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    # name = request.form['username']
    
    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    # Save the file
    filepath = os.path.join(app.config.root_path +'/' 'upload_folder', file.filename)
    file.save(filepath)
    name = filepath
    # return render_template('result.html', name=name)
    # import image_class_model
    model = load_model("image_class_model.h5")
    img = name
    img_width = 192
    img_height = 192
    image = keras.utils.load_img(img, target_size=(img_width, img_height))  
    img_arr = keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr, axis=0)  # Add batch dimension
    predictions = model.predict(img_bat)
    data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
    score = tf.nn.softmax(predictions)  
    # print('Veg/Fruit in image is {} with accurancy of {:0.2f}'.format(image_class_model.data_cat[np.argmax(score)],np.max(score)*100))
    return render_template('result.html', name='This image is '+data_cat[np.argmax(score)])

if __name__ == '__main__':
    app.run(debug=True)
    


# import image_class_model
# model =  image_class_model
# img = 'apple.jpg'
# img_width = 192
# img_height = 192
# image = keras.utils.load_img(img, target_size=(img_width, img_height))  
# img_arr = keras.utils.array_to_img(image)
# img_bat = tf.expand_dims(img_arr, axis=0)  # Add batch dimension
# predictions = model.model.predict(img_bat)

# score = tf.nn.softmax(predictions)  
# print('Veg/Fruit in image is {} with accurancy of {:0.2f}'.format(image_class_model.data_cat[np.argmax(score)],np.max(score)*100))