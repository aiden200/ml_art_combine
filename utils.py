import tensorflow as tf
import numpy as np
from PIL import Image


def J_content(h_C, h_G):
    # print(h_C)
    h_C = h_C[-1]
    h_G = h_G[-1]
    _, h_H, h_W, h_D = h_C.get_shape().as_list()
    content_cost = tf.reduce_sum(tf.square(tf.subtract(h_C, h_G)))
    return content_cost/(4*h_H*h_W*h_D)

def J_style(h_S, h_G):
    _, h_H, h_W, h_D = h_S.get_shape().as_list()
    h_S = tf.transpose(tf.reshape(h_S, shape=(h_H*h_W, h_D)))
    h_G = tf.transpose(tf.reshape(h_G, shape=(h_H*h_W, h_D)))
    G_hs = tf.matmul(h_S, tf.transpose(h_S))
    G_hg = tf.matmul(h_G, tf.transpose(h_G))
    return tf.reduce_sum(tf.square(tf.subtract(G_hs, G_hg)))/(4*(h_D**2)*((h_W*h_H)**2))

def total_J_style(S_I, G_I, layer_weights):

    S_I = S_I[:-1]
    G_I = G_I[:-1]
    total_cost = 0
    for weight, i in zip(layer_weights, range(len(S_I))):
        total_cost += weight[1]*J_style(S_I[i], G_I[i])
    return total_cost

def total_cost_function(h_C, S_I, G_I, layer_weights, alpha = 10, beta = 40):
    return alpha*J_content(h_C, G_I) + beta*total_J_style(S_I, G_I, layer_weights)


def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)