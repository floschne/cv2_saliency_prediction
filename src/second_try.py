import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import cv2

batch_size = 16

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return np.asarray(images).astype('float32') / 255.0


train_images = load_images_from_folder('cv2_data/train/images')
train_images_fixation = load_images_from_folder('cv2_data/train/fixations')
val_images = load_images_from_folder('cv2_data/val/images')
val_images_fixation = load_images_from_folder('cv2_data/val/fixations')
test_images = load_images_from_folder('cv2_data/test/images')

# train_images = train_images.astype('float32') 
# train_images_fixation = train_images_fixation.astype('float32') 
# val_images = val_images.astype('float32') 
# val_images_fixation = val_images_fixation.astype('float32')
# test_images = test_images.astype('float32')
# 
# train_images /= 255
# train_images_fixation /= 255
# val_images /= 255
# val_images_fixation /= 255
# test_images /= 255

input_shape = train_images[0].shape

checkpoint_path_old = "training_300_epochs/cp.ckpt"
checkpoint_path = "training_300_epochs/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=0)

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
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

# forth layer conv_4 24 x 40
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

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
              optimizer=keras.optimizers.SGD(lr=0.001))

#model.load_weights(checkpoint_path_old)

#model.fit(train_images, train_images_fixation, batch_size=batch_size, epochs=50,
#        verbose=1, validation_data=(val_images, val_images_fixation),
#        callbacks = [cp_callback])

model.load_weights(checkpoint_path)
predictions = model.predict(test_images, batch_size=batch_size, verbose=1)

#print(model.summary())

for idx, val in enumerate(test_images):
    #plt.imshow(predictions[idx, :, :, 0], cmap='gray')
    f, axarr = plt.subplots(2)
    axarr[0].imshow(test_images[idx])
    axarr[1].imshow(predictions[idx, :, :, 0], cmap='gray')

    #axarr[0].imshow(train_images[idx])
    #axarr[1].imshow(train_images_fixation[idx])
    #axarr[2].imshow(predictions[idx, :, :, 0], cmap='gray')

    plt.savefig('test_300epochs_predictions_{}.png'.format(idx))
    plt.close()
    #plt.show()
