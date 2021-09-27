import keras
import numpy as np
from keras.utils.np_utils import to_categorical
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=0.1, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_batch_end(self, batch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            print('stopping with %f'%current)
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
        if current < self.value:
            print('stopping with %f'%current)
            self.model.stop_training = True
        if current > 7.00:
            self.model.stop_training = True

class TextBoxCNN:
    def __init__(self, lr=0.0001, batch=128, max_epoch=10, interm_dim=200, input_shape=(64, 64, 3), model_name='w.pt', output_dir='../outputs'):
        self.input_shape = input_shape
        self.model = keras.Sequential([
            #keras.layers.InputLayer(input_shape=(64, 64, 3)),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', input_shape=input_shape),
            #keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(interm_dim, activation='relu'),
            keras.layers.Dense(2)
        ])
        self.lr = lr
        self.opt = keras.optimizers.Adam(lr=self.lr)
        #self.opt = keras.optimizers.SGD(lr=self.lr)
        self.batch = batch
        self.max_epoch = max_epoch
        self.model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])
        self.modelfile = os.path.join(output_dir, model_name)

    def train(self, x_train, y_train, retrain=False, validate=False, earlystop=False, verbose=True):
        if earlystop:
            #cb = [EarlyStopping(monitor='accuracy', mode='min', verbose=1)]
            #cb = [EarlyStopping(monitor='loss', patience=1, mode='min'), ModelCheckpoint(self.modelfile, monitor='loss', mode='min', save_best_only=True)]
            cb = [EarlyStoppingByLossVal(monitor='loss', value=0.01)]
        else:
            cb = []
        if os.path.exists(self.modelfile) and not retrain:
            self.model.load_weights(self.modelfile)
        elif not os.path.exists(self.modelfile) and retrain:
            raise ValueError('modelfile not found')
        else:
            y_train_oh = to_categorical(y_train, 2)
            if validate:
                self.model.fit(x_train, y_train_oh, batch_size=self.batch, epochs=self.max_epoch, validation_split=0.1, shuffle=True, callbacks=cb)
            else:
                self.model.fit(x_train, y_train_oh, batch_size=self.batch, epochs=self.max_epoch, validation_split=0, shuffle=True, callbacks=cb)
            self.model.save_weights(self.modelfile)
        if verbose:
            print(self.model.summary())

    def test(self, x_test, y_test):
        pred = self.model.predict_classes(x_test)
        score = (pred == y_test).sum() / y_test.shape[0]
        print('Accuracy=%f'%score)
        return score

class TextBoxCNN_adv(TextBoxCNN):
    def __init__(self, lr=0.0001, batch=128, max_epoch=10, interm_dim=200, input_shape=(64, 64, 3), model_name='w.pt', output_dir='../outputs'):
        self.input_shape = input_shape
        self.model = keras.Sequential([
            #keras.layers.InputLayer(input_shape=(64, 64, 3)),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', input_shape=input_shape),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(interm_dim, activation='relu'),
            keras.layers.Dense(interm_dim, activation='relu'),
            keras.layers.Dense(2)
        ])
        #self.model = VGG()
        self.lr = lr
        self.opt = keras.optimizers.Adam(lr=self.lr)
        #self.opt = keras.optimizers.SGD(lr=self.lr)
        self.batch = batch
        self.max_epoch = max_epoch
        self.model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])
        self.modelfile = os.path.join(output_dir, model_name)