import os 
import sys
import tensorflow as tf
import numpy as np
from test_functions import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from utils import *
import argparse


@tf.function()
def train_step(generated_image, vgg_model_outputs, h_C, S_I, layer_weights, optimizer):
    with tf.GradientTape() as tape:
        G_I = vgg_model_outputs(generated_image)
        cost = total_cost_function(h_C, S_I, G_I, layer_weights)
                
    grad = tape.gradient(cost, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return cost


def load_and_train_models(content_file, style_file, img_size=800):
    #load pre trained models
    vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained_models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False 


    content_image = np.array(Image.open(content_file).resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    style_image =  np.array(Image.open(style_file).resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    if content_image.shape[-1] == 4:
        content_image = content_image[:,:,:-1]
    if style_image.shape[-1] == 4:
        style_image = style_image[:,:,:-1]

    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.5, 0.5)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    content_layer = [('block5_conv4', 1)]
    layer_weights = [('block1_conv1', 0.1), ('block2_conv1', 0.2), ('block3_conv1', 0.4), ('block4_conv1', 0.2), ('block5_conv1', 0.1)]

    vgg_model_outputs = get_layer_outputs(vgg, layer_weights + content_layer)

    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    h_C = vgg_model_outputs(preprocessed_content)   

    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    h_S = vgg_model_outputs(preprocessed_style)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    generated_image = tf.Variable(generated_image)
    # print(f"type: {type(h_C)}")
    epochs = 2501
    for i in range(epochs):
        train_step(generated_image, vgg_model_outputs, h_C, h_S, layer_weights, optimizer)
        if i % 50 == 0:
            print(f"Epoch {i} ")
        if i % 500 == 0:
            image = tensor_to_image(generated_image)
            imshow(image)
            image.save(f"output/image_{i}.jpg")
            plt.show() 


def run_model():
    parser = argparse.ArgumentParser(description='Choose File')
    parser.add_argument("-c", "--content", help="content file")
    parser.add_argument("-s", "--style", help="style file")
    args = parser.parse_args()

    content_file = "content/content.jpeg",
    style_file = "style/drop-of-water.jpg"
    img_size = 800
    if args.c:
        content_file = args.c
    if args.s:
        style_file = args.s
    
    load_and_train_models(content_file, style_file, img_size)



def test_functions():
    J_content_test(J_content)
    J_style_test(J_style)


if __name__ == "__main__":
    run_model()