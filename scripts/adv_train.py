"""
This is an example of how to use ART and Keras to perform adversarial training using data generators for CIFAR10
"""
import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
import os
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, ZeroPadding2D
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

import sys
sys.path.append('../')
import smerf
import argparse

from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.classifiers import KerasClassifier
from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainer
import time

# Directory to save the data
DATA_DIR = '../data'

# Directory to save intermediary/final results
CACHE_DIR = '../outputs/cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
    
# Directory to save plots
PLOT_DIR = '../outputs/plots'
if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)
    
def build_simple_model(input_shape=(64,64,3), interm_dim=200, lr=0.0001):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(interm_dim, activation='relu'),
        keras.layers.Dense(2)
    ])
    opt = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_complex_model(input_shape=(64,64,3), interm_dim=200, lr=0.0001):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(interm_dim, activation='relu'),
        keras.layers.Dense(interm_dim, activation='relu'),
        keras.layers.Dense(2)
    ])
    opt = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    exp_no = args.exp # experiment number
    attack_type = args.attack # attack type
    epochs = args.ep
    batch_size = args.batch
    print('EXP_NO = %f'%exp_no)
    lr = args.lr # learning rate used to train the model
    model_name = 'w-adv2-%0.2f.pt'%exp_no # model file to save/load
    train_type = args.type
    
    if exp_no == 1.11: 
        import smerf.simple_fr as textbox_exp
        no_data = 2000
    elif exp_no == 2.11: 
        import smerf.simple_nr as textbox_exp
        no_data = 5000
    elif exp_no == 1.2:
        import smerf.complex_fr as textbox_exp
        no_data = 2000
    elif exp_no == 3.71: # Complex-CR1
        import smerf.complex_cr1 as textbox_exp
        no_data = 15000
    elif exp_no == 3.72: # Complex-CR2
        import smerf.complex_cr2 as textbox_exp
        no_data = 15000
    elif exp_no == 3.73: # Complex-CR3
        import smerf.complex_cr3 as textbox_exp
        no_data = 15000
    elif exp_no == 3.74: # Complex-CR4
        import smerf.complex_cr4 as textbox_exp
        no_data = 15000
    
    ### Generate (or load) datasets        
    train_data, test_data, train_primary, test_primary, train_secondary, test_secondary = \
                textbox_exp.generate_textbox_data(n=no_data, 
                                                  save=True, 
                                                  save_dir='../data', 
                                                  exp_no=exp_no,
                                                  random_bg=0)       
    x_train = train_data.X
    x_test = test_data.X
    y_train = train_data.y
    y_test = test_data.y
    
    original_name = 'w%0.2f.pt'%exp_no

    y_train_oh = to_categorical(y_train, 2)
    y_test_oh = to_categorical(y_test, 2)
    
    print('data loaded')
    
    datagen = ImageDataGenerator()
    datagen.fit(x_train)
    art_datagen = KerasDataGenerator(datagen.flow(x=x_train, 
                                                  y=y_train_oh, 
                                                  batch_size=batch_size, 
                                                  shuffle=True),
                                    size=x_train.shape[0], batch_size=batch_size)
    print('generator fit')

    if exp_no == 1.11 or exp_no == 2.11:
        model = build_simple_model(lr=lr)
        #if train_type != 'scratch':
        model.load_weights(os.path.join(CACHE_DIR, original_name))
        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
    else:
        model = build_complex_model(lr=lr)
        #if train_type != 'scratch':
        model.load_weights(os.path.join(CACHE_DIR, original_name))
        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
            
    # Create attack for adversarial trainer; here, we use 2 attacks, both crafting adv examples on the target model
    print('Creating Attack')
    if attack_type == 'pgd':
        attacker = ProjectedGradientDescent(classifier, eps=0.3, eps_step=0.1, max_iter=10, num_random_init=1)  
    elif attack_type == 'fgsm':
        attacker = FastGradientMethod(classifier, eps=0.2)
    else:
        raise ValueError()
    
    # Create advareasial samples
    if os.path.exists('x_train2_%s_%0.2f.npy'%(attack_type, exp_no)):
        x_train_pgd = np.load('x_train2_%s_%0.2f.npy'%(attack_type, exp_no))
        x_test_pgd = np.load('x_test2_%s_%0.2f.npy'%(attack_type, exp_no))
    else:
        x_test_pgd = attacker.generate(x_test)
        x_train_pgd = attacker.generate(x_train)
        np.save(open('x_train2_%s_%0.2f.npy'%(attack_type, exp_no), 'wb'), x_train_pgd)
        np.save(open('x_test2_%s_%0.2f.npy'%(attack_type, exp_no), 'wb'), x_test_pgd)
    print(x_test_pgd.shape)
    

    preds = np.argmax(classifier.predict(x_test_pgd), axis=1)
    acc = np.sum(preds == np.argmax(y_test_oh, axis=1)) / y_test.shape[0]
    print("Classifier before adversarial training")
    print("Accuracy on adversarial samples: %.2f%%", (acc * 100))
    
    # Create adversarial trainer and perform adversarial training
    print('Training')

    if train_type == 'aug':
        # augment the data
        x_train = np.append(x_train, x_train_pgd, axis=0)
        y_train = np.append(y_train, y_train, axis=0)
        y_train_oh = np.append(y_train_oh, y_train_oh, axis=0)

        #retrain the model with augmented data
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(x_train, y_train_oh, nb_epochs=epochs, batch_size=batch_size, verbose=True)
        classifier._model.save_weights(os.path.join(CACHE_DIR, model_name))
        
    elif train_type == 'scratch':
        adv_trainer = AdversarialTrainer(classifier, attacks=attacker, ratio=1.)
        adv_trainer.fit_generator(art_datagen, nb_epochs=epochs)
        model = classifier._model
        model.save_weights(os.path.join(CACHE_DIR, model_name)) 

    # load the pretrained model
    #model.load_weights(os.path.join(CACHE_DIR, model_name))
    #classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
    # adv_trainer = AdversarialTrainer(classifier, attacks=attacker, ratio=1.)
    # adv_trainer.fit_generator(art_datagen, nb_epochs=20)
    # model = classifier._model
    # model.save_weights(os.path.join(CACHE_DIR, model_name))

    # Evaluate the adversarially trained model on clean test set
    labels_true = np.argmax(y_test_oh, axis=1)
    labels_test = np.argmax(classifier.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))
    
    # Evaluate the adversarially trained model on original adversarial samples
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on original PGD adversarial samples: %.2f%%' % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100)) 
    
    # Evaluate the adversarially trained model on fresh adversarial samples produced on the adversarially trained model
    x_test_pgd = attacker.generate(x_test)
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on new PGD adversarial samples: %.2f%%' % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=float, help='experiment number. 1-FR, 2-NR, 3-CR')
    parser.add_argument('--ep', type=int, default=10, help='max epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--attack', type=str, default='fgsm', help='attack method')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--type', type=str, default='scratch', help='train type')
    args = parser.parse_args()
    print(args)
    main(args)
