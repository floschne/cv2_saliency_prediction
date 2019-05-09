import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
from matplotlib import pyplot as plt
import cv2


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


train_images = load_images_from_folder('cv2_data/train/images')
train_images_fixation = load_images_from_folder('cv2_data/train/fixations')

val_images = load_images_from_folder('cv2_data/val/images')
val_images_fixation = load_images_from_folder('cv2_data/val/fixations')

test_images = load_images_from_folder('cv2_data/test/images')

print(len(train_images))
print(train_images_fixation[0].shape)
print(train_images_fixation[0].dtype)

input_shape = train_images[0].shape