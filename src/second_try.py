import datetime
import os

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return np.asarray(images).astype('float32') / 255.0


@click.command()
@click.option('--operation', '-o', type=click.Choice(['train', 'predict', 'eval']),
              help='The operation to perform')
@click.option('--batch-size', '-bs', default=8, type=int, help='The batch size for training.', show_default=True)
@click.option('--train-size', '-ts', default=None, type=int, help='The training set size. Default is all images.')
@click.option('--epochs', '-e', default=1, help='The number of passes over the whole training set when training.',
              show_default=True)
@click.option('--data-dir', '-dd', default="./cv2_data/", type=str, help='Where to store or load the data.')
@click.option('--model-dir', '-md', default="./model/", type=str, help='Where to store or load model data.')
@click.option('--checkpoint-path', '-cpp', default="./model_checkpoints/cp.ckpt", type=str,
              help='Where to store or load checkpoints of the model.')
def main(operation, batch_size, train_size, epochs, data_dir, model_dir, checkpoint_path):
    # Verify that the there was at least one operation requested
    if not operation:
        print('No options given, try invoking the command with "--help" for help.')

    # TODO do we need all of this?!
    # checkpoint_path_old = "training_300_epochs/cp.ckpt"
    # checkpoint_path = "training_300_epochs/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    if operation == "train":

        train_images = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'train/images'))
        train_images_fixation = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'train/fixations'))
        val_images = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'val/images'))
        val_images_fixation = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'val/fixations'))

        print(train_images.shape)
        print(train_images_fixation[0].shape)
        print(train_images[50:60].shape)

        input_shape = train_images[0].shape
        print(input_shape)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0)
        # Create Tensorboard callback
        logdir = "./logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        model = keras.models.Sequential()
        # first layer conv_1 180 x 320 #-> 192 x 320
        model.add(keras.layers.ZeroPadding2D(padding=(6, 0)))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                                      data_format='channels_last',
                                      input_shape=input_shape))
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

        # TODO why this?
        # model.load_weights(checkpoint_path)

        model.fit(train_images, train_images_fixation,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(val_images, val_images_fixation),
                  callbacks=[cp_callback, tensorboard_callback])

        print("Serializing model to %s" % os.path.abspath(model_dir))
        # serialize model to json
        model_json = model.to_json()
        with open(os.path.join(os.path.abspath(model_dir), 'model.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5 TODO I guess this is now obsolete since we're using checkpoints
        # model.save_weights(os.path.join(os.path.abspath(model_dir), 'model.h5'))

    elif operation == "predict":
        test_images = load_images_from_folder('cv2_data/test/images')
        print(test_images.shape)

        # deserialize model from json
        json_file = open(os.path.join(os.path.abspath(model_dir), 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into deserialized model
        # loaded_model.load_weights(os.path.join(os.path.abspath(model_dir), 'model.h5'))
        loaded_model.load_weights(checkpoint_path)
        print("Loaded model from %s" % os.path.abspath(model_dir))

        predictions = loaded_model.predict(test_images, batch_size=batch_size, verbose=1)
        print(predictions.shape)

        for idx, val in enumerate(test_images):
            # plt.imshow(predictions[idx, :, :, 0], cmap='gray')
            f, axarr = plt.subplots(2)
            axarr[0].imshow(test_images[idx])
            axarr[1].imshow(predictions[idx, :, :, 0], cmap='gray')

            # axarr[0].imshow(train_images[idx])
            # axarr[1].imshow(train_images_fixation[idx])
            # axarr[2].imshow(predictions[idx, :, :, 0], cmap='gray')

            plt.savefig('test_300epochs_predictions_{}.png'.format(idx))
            plt.close()
            # plt.show()

    elif operation == "eval":
        raise NotImplemented("Not yet implemented!")


if __name__ == '__main__':
    main()
