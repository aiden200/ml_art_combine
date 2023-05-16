import tensorflow as tf
import numpy as np


def J_content_test(target):

    tf.random.set_seed(10)
    a_C = tf.random.normal([4, 4, 3], mean=1, stddev=4)
    a_G = tf.random.normal([4, 4, 3], mean=1, stddev=4)
    J_content = target(a_C, a_G)

    assert np.isclose(J_content, 9.546153), f"Expected 9.546153, got {J_content}"
    print("J_content_cost = " + str(J_content))
    print("\033[92mAll tests passed")



        
    
def J_style_test(target):
    tf.random.set_seed(10)
    a_S = tf.random.normal([4, 4, 3], mean=1, stddev=4)
    a_G = tf.random.normal([4, 4, 3], mean=1, stddev=4)
    J_style_layer_SG = target(a_S, a_G)

    assert J_style_layer_SG > 0, "Wrong value. compute_layer_style_cost(A, B) must be greater than 0 if A != B"
    assert np.isclose(J_style_layer_SG, 6.309712), "Wrong value."
    print("J_style_cost = " + str(J_style_layer_SG))
    print("\033[92mAll tests passed")
    
### ex 4 is already implemented for the learners


### ex 5
def total_cost_test(target):
    J_content = 0.2    
    J_style = 0.8
    J = target(J_content, J_style)

    assert type(J) == EagerTensor, "Do not remove the @tf.function() modifier from the function"
    assert J == 34, "Wrong value. Try inverting the order of alpha and beta in the J calculation"
    assert np.isclose(target(0.3, 0.5, 3, 8), 4.9), "Wrong value. Use the alpha and beta parameters"

    np.random.seed(1)
    print("J = " + str(target(np.random.uniform(0, 1), np.random.uniform(0, 1))))

    print("\033[92mAll tests passed")
    

### ex 6
def train_step_test(target, generated_image):
    generated_image = tf.Variable(generated_image)


    J1 = target(generated_image)
    print(J1)
    assert type(J1) == EagerTensor, f"Wrong type {type(J1)} != {EagerTensor}"
    assert np.isclose(J1, 25629.055, rtol=0.05), f"Unexpected cost for epoch 0: {J1} != {25629.055}"

    J2 = target(generated_image)
    print(J2)
    assert np.isclose(J2, 17812.627, rtol=0.05), f"Unexpected cost for epoch 1: {J2} != {17735.512}"

    print("\033[92mAll tests passed")
    
    
    
    
    
    