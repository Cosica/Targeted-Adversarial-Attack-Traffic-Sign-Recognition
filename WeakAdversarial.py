import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import tensorflow as tf
import torch

from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

def preprocess_image(filename):
    img_width, img_height = 224, 224
    image = load_img(filename, target_size=(img_width, img_height))
    image = img_to_array(image)/255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.convert_to_tensor(image)
    return image

def get_gradients(image, labels,model):
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_function(labels, prediction)
    gradients = tape.gradient(loss, image)
    return gradients

def creat_adversarial_example(model, img_path, target):
    alpha = 1
    epsilon = 3 / 255
    steps = 10    
    perturbation = np.full((1, 224, 224, 3), 0)
    image_init = preprocess_image(img_path)
    plt.imsave("adv_img.png", image_init[0].numpy())
    y_LL = model.predict(image_init)
    index = np.argmax(y_LL)
    temp = y_LL[0][index]-0.1
    y_LL[0][index] = y_LL[0][target]
    y_LL[0][target] = temp
    for i in range(3):    
        image_var = preprocess_image("adv_img.png")
        x_adv = tf.Variable(image_var)
        for i in range(steps):
            gradients = get_gradients(x_adv, y_LL,model)
            below = x_adv - epsilon
            above = x_adv + epsilon
            x_adv = x_adv - alpha * tf.sign(gradients)
            x_adv = tf.clip_by_value(tf.clip_by_value(x_adv, below, above), 0, 1)
        plt.imsave("adv_img.png", x_adv[0].numpy())
    return x_adv.numpy()

def main(model,img_path,target):
    x = creat_adversarial_example(model,img_path,target)
    return x

if __name__=="__main__":
    main(model,img_path,target)