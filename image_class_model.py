import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, models


data_train_path = '/home/yhs/Documents/AI_RIG/AI Project/Fruits_Vegetables/train'
data_test_path = '/home/yhs/Documents/AI_RIG/AI Project/Fruits_Vegetables/test'
data_val_path = '/home/yhs/Documents/AI_RIG/AI Project/Fruits_Vegetables/validation'

img_width = 192
img_height = 192

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)

data_cat = data_train.class_names

data_val = tf.keras.utils.image_dataset_from_directory(data_val_path,
    shuffle=False,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)

data_test = tf.keras.utils.image_dataset_from_directory(data_test_path,
    shuffle=False,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)

plt.figure(figsize=(10, 10))
for image, labels in data_train.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].numpy().astype("uint8"))
        plt.title(data_cat[labels[i]])
        
        
from keras.models import Sequential

data_train


model = Sequential([
    layers.Rescaling(1./255),#Purpose: Normalize pixel values from [0, 255] → [0, 1]. Neural networks learn faster and better when inputs are normalized.
    layers.Conv2D(16,3, padding='same',activation='relu'),#convolution make image to 
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128,activation='relu'),
    layers.Dense(len(data_cat),activation='softmax')
])


# optimizer_t = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
epochs_size = 12
history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

epochs_range = range(epochs_size)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,history.history['accuracy'],label = 'Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'],label = 'Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['loss'],label = 'Training Loss')
plt.plot(epochs_range, history.history['val_loss'],label = 'Validation Loss')
plt.title('Loss')

model.save('image_class_model.h5')