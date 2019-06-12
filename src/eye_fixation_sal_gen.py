import datetime
import os

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from util import KLD
from util import load_images_from_folder
from util import load_model


@click.command()
@click.option('--operation', '-o', type=click.Choice(['train', 'predict', 'eval']),
              help='The operation to perform')
@click.option('--batch-size', '-bs', default=4, type=int, help='The batch size for training.', show_default=True)
@click.option('--epochs', '-e', default=1, help='The number of passes over the whole training set when training.',
              show_default=True)
@click.option('--data-dir', '-dd', default="./cv2_data/", type=str, help='Where to store or load the data.')
@click.option('--model-dir', '-md', default="./model/", type=str, help='Where to store or load model data.')
@click.option('--log-dir', '-ld', default="./logs/", type=str, help='Where to store the logs.')
@click.option('--pred-dir', '-pd', default="./preds/", type=str, help='Where to store the predictions.')
@click.option('--generate-final-predictions', default=False, type=bool,
              help='If true predictions for final evaluation are generated or else for visual human evaluation.')
@click.option('--checkpoint-path', '-cpp', default="./model_checkpoints/cp.ckpt", type=str,
              help='Where to store or load checkpoints of the model.')
def main(operation, batch_size, epochs, data_dir, model_dir, log_dir, pred_dir, generate_final_predictions,
         checkpoint_path):
    # Verify that the there was at least one operation requested
    if not operation:
        print('No options given, try invoking the command with "--help" for help.')

    # traning the model
    if operation == "train":
        # loading the traning data
        train_images = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'train/images'))
        train_images_fixation = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'train/fixations'))
        val_images = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'val/images'))
        val_images_fixation = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'val/fixations'))

        print("train_images.shape: " + str(train_images.shape))
        print("train_images_fixation[0].shape: " + str(train_images_fixation[0].shape))

        input_shape = train_images[0].shape
        print("train_images[0].shape: " + str(train_images[0].shape))

        # Create checkpoint callback

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=False,
                                                         period=50,
                                                         verbose=0)
        # Create Tensorboard callback
        logdir = os.path.join(os.path.abspath(log_dir), str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        if not os.path.exists(logdir):
            os.makedirs(os.path.dirname(logdir))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        # The generator Model as it is presented in the Salgan network 
        model = keras.models.Sequential()
        # first layer conv_1 180 x 320 #-> 192 x 320, zero padding is needed to achieve max pooling depth
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

        # compiing the model
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.001))

        # train the model on the click parameters that were specified
        model.fit(train_images, train_images_fixation,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(val_images, val_images_fixation),
                  callbacks=[cp_callback, tensorboard_callback])

        print("Serializing model to %s" % os.path.abspath(model_dir))
        # serialize model to json
        model_json = model.to_json()
        if not os.path.exists(os.path.dirname(model_dir)):
            os.makedirs(os.path.dirname(model_dir))
        f = open(os.path.join(os.path.abspath(model_dir), "model.json"), "w")
        f.write(model_json)
        f.close()

    elif operation == "predict":
        # load the test images
        test_images = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'test/images'))

        loaded_model = load_model(checkpoint_path, model_dir)

        # predicting the test data
        print("Predicting eye fixation maps...")
        predictions = loaded_model.predict(test_images, batch_size=batch_size, verbose=1)

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        # output on predictions
        print("Writing files...")
        for idx, val in enumerate(predictions):
            if not generate_final_predictions:
                f, axarr = plt.subplots(2)
                axarr[0].imshow(test_images[idx], cmap='gray')
                axarr[1].imshow(predictions[idx, :, :, 0], cmap='gray')

                # saving output to file
                plt.savefig(os.path.join(os.path.abspath(pred_dir), 'test_predictions_{}.png'.format(1600 + idx + 1)))
                plt.close()
            else:
                img = predictions[idx]
                img *= 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(os.path.join(os.path.abspath(pred_dir), '{}_prediction.jpg'.format(1600 + idx + 1)), img)

    elif operation == "eval":
        # load model
        loaded_model = load_model(checkpoint_path, model_dir)

        # load the val images
        val_images = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'val/images'))
        # load the ground truth images
        labels = load_images_from_folder(os.path.join(os.path.abspath(data_dir), 'val/fixations'))
        # predict the fixations from val images
        print("Predicting eye fixation maps...")
        preds = loaded_model.predict(val_images, batch_size=batch_size, verbose=1)

        assert len(labels) == len(preds), "Labels and predictions have to have the same number of elements!"

        kld_avg = 0
        for (p, g) in zip(preds, labels):
            kld_avg += KLD(p[:, :, 0], g[:, :, 0])

        kld_avg /= len(labels)
        print('Average KLD for validation data set: {:f}'.format(kld_avg))


if __name__ == '__main__':
    main()
