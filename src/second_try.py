import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
from matplotlib import pyplot as plt
import cv2
import math

batch_size = 4


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return np.asarray(images)


train_images = load_images_from_folder('cv2_data/train/images')[:120]
train_images_fixation = load_images_from_folder('cv2_data/train/fixations')[:120]

val_images = load_images_from_folder('cv2_data/val/images')[:60]
val_images_fixation = load_images_from_folder('cv2_data/val/fixations')[:60]

test_images = load_images_from_folder('cv2_data/test/images')[:10]

print(train_images.shape)
print(train_images_fixation[0].shape)
print(test_images.shape)
print(train_images[50:60].shape)

input_shape = train_images[0].shape
print(input_shape)

model = keras.models.Sequential()

# first layer conv_1 180 x 320 #-> 192 x 320
model.add(keras.layers.ZeroPadding2D(padding=(6, 0)))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', data_format='channels_last', input_shape=input_shape))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

# second layer conv_2 96 x 160
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

# third layer conv_3 48 x 80
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

# forth layer conv_4 24 x 40
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

# fifth layer conv_5 12 x 20
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))

# sixth layer 'reverse' conv_6 12 x 20
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.UpSampling2D(size=(2, 2)))

# seventh layer 'reverse' conv_7 24 x 40
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.UpSampling2D(size=(2, 2)))

# eighth layer 'reverse' conv_8 48 x 80
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.UpSampling2D(size=(2, 2)))

# ninth layer 'reverse' conv_9 96 x 160
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.UpSampling2D(size=(2, 2)))

# tenth layer 'reverse' conv_10 192 x 320
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

# output layer 192 x 320 -> 180 x 320 x 1
model.add(keras.layers.Cropping2D(cropping=(6, 0)))
model.add(keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(train_images, train_images_fixation, batch_size=batch_size, epochs=1,
          verbose=1, validation_data=(val_images, val_images_fixation))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

predictions = model.predict(test_images, batch_size=batch_size, verbose=1)

print(predictions.shape)
#print(model.summary())

plt.imshow(train_images[50])
plt.show()

plt.imshow(predictions[0, :, :, 0], cmap='gray')
plt.show()

plt.imshow(train_images_fixation[50], cmap='gray')
plt.show()

