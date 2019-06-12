import os

import cv2
import numpy as np
from scipy.special import softmax
from tensorflow import keras


def load_model(checkpoint_path, model_dir):
    # deserialize model from json
    json_file = open(os.path.join(os.path.abspath(model_dir), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into deserialized model
    loaded_model.load_weights(checkpoint_path)
    print("Loaded model from %s" % os.path.abspath(model_dir))
    return loaded_model


# load image data convert it to float and normalize the input to be between 0 and 1
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    print("Loaded {:d} images from {:s}".format(len(images), folder))
    return np.asarray(images).astype('float32') / 255.0


# Input: two 2D floating point numpy arrays of the same shape.
# The inputs will be interpreted as probability mass functions and the KL divergence is returned.
def KLD(P, G):
    if P.ndim != 2:
        raise ValueError("Expected P to be 2 dimensional array")
    if G.ndim != 2:
        raise ValueError("Expected G to be 2 dimensional array")
    if P.shape != G.shape:
        raise ValueError('The shape of P: {} must match the shape of G: {}'.format(P.shape, G.shape))
    if np.any(P < 0):
        raise ValueError('P has some negative values')
    if np.any(G < 0):
        raise ValueError('G has some negative values')

    # Normalize P and G using softmax
    p_n = softmax(P)
    g_n = softmax(G)

    EPS = 1e-16  # small regularization constant for numerical stability
    kl = np.sum(g_n * np.log2(EPS + (g_n / (EPS + p_n))))

    return kl
