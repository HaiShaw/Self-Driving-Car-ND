# Henry.X Udacity
import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.image as pimg
from sklearn.model_selection import train_test_split
from random import random
from PIL import Image
import os.path
import json

from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Input, MaxPooling2D, Convolution2D, Dropout, Activation, Flatten, Dense
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras import initializations
from keras import backend as K

# My Keras Helpers:
#
# have not found truncated_normal init. from Keras stocked initializations
# so come up one own to experient:
# equivalent to 0 mean tf.truncated_normal, except that it return np array

def trunc_norm(shape, stddev=0.1):
    norm = np.random.normal(scale=stddev, size=shape)
    trun = np.clip(norm, -2.0*stddev, 2.0*stddev)
    return trun

# N.B.
#     he_normal() is an overloaded function
#     it is used to initialize very tiny model weights similar to tf.truncated_normal()
#
# Unable to load custom initializer from the saved model, passing custom_objects is not working
# so have to overload an existing initializations (we overload 'he_normal' as truncated_normal)
# https://github.com/fchollet/keras/issues/3867
# https://github.com/fchollet/keras/issues/1634

def he_normal(shape, name=None):
    value = trunc_norm(shape, stddev=0.001)
    return K.variable(value, name=name)


train_file =  'train.pkl'
test_file = 'udacity.pkl'
# Pickled Udacity data the same way below!

##########################################
# pickle training data set if not done yet
#
if (not os.path.exists(train_file)):

    if (not os.path.exists('driving_log.csv')):
        exit("Rerun with driving_log.csv or pre-pickled train.pkl in this path!")

    ##############################
    # READ drive data to DataFrame
    #
    # Read driving_log.csv to pandas data frame
    # N.B. driving_log.csv HAS NO headers given!
    # So I added column names.
    df = pd.read_csv('driving_log.csv', header=None, usecols=[0,1,2,3], names=['center', 'left', 'right', 'angle'])

    #####################
    # Input preprocessing
    #
    # data input preprocess, pickle images and steering angles first
    # N.B. I didn't need huge training set to generate final model, so I did not use Keras generator yet here.
    #      My environment: Macbook Pro, 16GB memory (but I will add generator use in the future)

    train = {}

    img = Image.open(df['center'][0])

    # RESIZE image (choose to half by W and H)
    # 1. better applicable to train model on CPU
    # 2. better fit to machine with memory limit
    # Shrink it by half. More shrink is possible.
    img = img.resize((160, 80), Image.ANTIALIAS)
    imga = np.asarray(img)
    img_na = [imga]

    angle = df['angle'][0]
    ang_na = [angle]

    # I used all three camera images to train the model, to acquire more samples automatically
    # tried 2 camera calibration scheme as below:
    # linear scheme turns out to be the fit one!
    linear = True
    if linear:
        for i in range(1, df.shape[0]):
            img_i = Image.open(df['center'][i])
            img_i = img_i.resize((160, 80), Image.ANTIALIAS)
            img_a = np.asarray(img_i)
            img_na = np.append(img_na, [img_a], axis=0)
            ang_i = df['angle'][i]
            ang_na = np.append(ang_na, [ang_i], axis=0)
       
            # Left Image & Steering adjusted (+0.2) 
            img_i = Image.open(df['left'][i].strip())
            img_i = img_i.resize((160, 80), Image.ANTIALIAS)
            img_a = np.asarray(img_i)
            img_na = np.append(img_na, [img_a], axis=0)
            ang_i = df['angle'][i] + 0.2 # ver: x
            ang_na = np.append(ang_na, [ang_i], axis=0)

            # Right Image & Steering adjusted (-0.2)
            img_i = Image.open(df['right'][i].strip())
            img_i = img_i.resize((160, 80), Image.ANTIALIAS)
            img_a = np.asarray(img_i)
            img_na = np.append(img_na, [img_a], axis=0)
            ang_i = df['angle'][i] - 0.2 # ver: x
            ang_na = np.append(ang_na, [ang_i], axis=0)
    else:
        # None linear camera image <-> steering angle calibration, did NOT choose all L/C/R
        for i in range(1, df.shape[0]):
            dice = random()
            ang_i = df['angle'][i]
            if dice > 0.66:
                img_i = Image.open(df['center'][i])
                img_i = img_i.resize((160, 80), Image.ANTIALIAS)
                img_a = np.asarray(img_i)
                img_na = np.append(img_na, [img_a], axis=0)
                ang_na = np.append(ang_na, [ang_i], axis=0)
            elif dice > 0.33:
                # Left Image & Steering adjusted to bring vehicle on track
                img_i = Image.open(df['left'][i].strip())
                img_i = img_i.resize((160, 80), Image.ANTIALIAS)
                img_a = np.asarray(img_i)
                img_na = np.append(img_na, [img_a], axis=0)
                if ang_i < -0.1:
                    # '-' left turn degree on left image 
                    ang_i *= 0.8
                elif ang_i > 0.1:
                    # '+' right turn degree on left image 
                    ang_i *= 1.2
                else:
                    # pick 0.1 as base calibrate (assume)
                    # Ignored theta vehicle axis to road center line
                    ang_i += 0.1
                ang_na = np.append(ang_na, [ang_i], axis=0)
            else:
                # Right Image & Steering adjusted to bring vehicle on track
                img_i = Image.open(df['right'][i].strip())
                img_i = img_i.resize((160, 80), Image.ANTIALIAS)
                img_a = np.asarray(img_i)
                img_na = np.append(img_na, [img_a], axis=0)
                if ang_i < -0.1:
                    # '+' left turn degree on right image 
                    ang_i *= 1.2
                    #ang_i -= 0.1
                elif ang_i > 0.1:
                    # '-' right turn degree on right image 
                    ang_i *= 0.8
                else:
                    # pick 0.1 as base calibrate
                    ang_i -= 0.1
                ang_na = np.append(ang_na, [ang_i], axis=0)
    
    train['images'] = img_na
    train['angles'] = ang_na

    # Pickling data set (for training and validation)
    with open('train.pkl', 'wb') as f:
        pickle.dump(train, f)

else:
  # Loading training data set if it is also available
  with open(train_file, mode='rb') as f:
      train = pickle.load(f)


#########################################
# Input normalization (same as project 2)
#

train['images'] = (train['images'] - 128.0)/128

X = train['images']
y = train['angles']


# Randomize output polarity for regression
# It has to come with image flip
# This image horizontal L/R Flip is needed.
for i in range(0, y.shape[0]):
    if np.random.choice([True, False]):
        X[i] = np.fliplr(X[i])
        y[i] = 0. - y[i]


# Train/validation/test split:
#
# Option 1. use Keras built-in validation split and shuffle in model.fit():
#   Keras validation_split always choose last portion (pertentage) of pre-shuffle data as validation set,
#   So it is okay if data set is shuffled prior to the split, otherwise the validation set given by this
#   built-in keras parameter would be very bias towards the last period of training records!
#   So use this option, if shuffle data set indices first (e.g. at pandas data frame loaded)
# Option 2. use sklearn train_test_split()
#
# Use option 2. here
#
# Split keep random 10% as validation set.
#
# Test set:
#   1. use udacity data set for testing - model.evaluate()
#   2. use simulator to test, and add recover/correction subsampling data when necessary
#
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=101)


###################
#
# Model definition:
# Model architecture summary:

# Network architecture (Preliminary choice: 2 ConvNets + 2 FullConnected NNets. Main constraints: Computational Power!)
# Shape of tensors passing along the graph is annotated in (); basic layers formation is annotated in []
# Input (,80,160,3)=> CNN1[3x5x5x24/(2x2 subsample)] =(,38,78,24)=> CNN2[24x5x5x32/(2x2 subsample)] =(,17,37,32)=> CNN3[32x5x5x48/(2x2 subsample)]
#      =( ,7,17,48)=> CNN4[48x3x3x64/(2x2subsample)] =(, 3, 8,64)=> CNN5[64x3x3x64] =(,1,6,64)=> Flatten =(,384)=> FC1[384x128] =(,128)=> Drop(.5)
#      => ELU() =(,128)=> FC2[128x64] =(,64)=> Dropout(.5) => ELU() =(,64)=> FC3[64x16] =(,16)=> Output[16x1] =(,1)=> Steering
# N.B.
#   Activation:
#   - CNN uses ReLU
#   - FC  uses ELU
#   - Output uses linear
#   Dropout/Regularization: 
#   - FC  uses Dropout(0.5), between FC1 and FC2
#   Others hyper parameters are tunable

model = Sequential()

# Originally tried shrink images from (160, 320, 3) to (80, 160, 3) using Keras layer - Maxpooling at Input
# It is replaced with img.resize() at preprocessing.
# model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(160, 320, 3), name='InMaxPool'))

model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, init='he_normal', activation='relu', subsample=(2, 2), name='CNN1', input_shape=(80, 160, 3)))
model.add(Convolution2D(nb_filter=32, nb_row=5, nb_col=5, init='he_normal', activation='relu', subsample=(2, 2), name='CNN2'))
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, init='he_normal', activation='relu', subsample=(2, 2), name='CNN3'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init='he_normal', activation='relu', subsample=(2, 2), name='CNN4'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init='he_normal', activation='relu', subsample=(1, 1), name='CNN5'))

model.add(Flatten(name='Flatten'))
model.add(Dense(128, init='he_normal', name='FC1'))
model.add(Dropout(0.5, name='DROP1'))
model.add(ELU())
model.add(Dense(64, init='he_normal', name='FC2'))
# model.add(Dropout(0.5, name='DROP2'))
model.add(ELU())
model.add(Dense(16, init='he_normal', name='FC3'))

# Linear activation fit well to output, steering angles are bewteen [-,+]
model.add(Dense(1, init='he_normal', name='Output'))

print("Network Summary:\n")
model.summary()
print("\n")

# N.B. accuracy is an improper metrics for this regression problem
#      may custom R^2 as model metrics later
adam = Adam(lr=0.00001)
model.compile(optimizer=adam, loss='mse')


#################
#
# Train the model
#
# N.B. Below save model at each epoch step, for testing in simulation!
epoch_nums = 20
batch_size = 100

print("Model Training:\n")
for i in range(epoch_nums):
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True, validation_data=(X_val, y_val))
    # Save model at each epoch
    model_ith_arch = 'model' + str(i) + '.json'
    model_ith_parm = 'model' + str(i) + '.h5'

    model_architecture = model.to_json()
    with open(model_ith_arch, 'w') as f:
        json.dump(model_architecture, f, ensure_ascii=False)

    model.save_weights(model_ith_parm)

print("\n")

##############################
#
# Load test data set (udacity)
#
# N.B. prepickled udacity data
#


if (not os.path.exists(test_file)):
    exit("To run Testing, rerun with pre-pickled udacity.pkl in this path!")

with open(test_file, mode='rb') as f:
    test = pickle.load(f)

# apply normalization
test['images'] = (test['images'] - 128.0)/128

# testing data
X_test = test['images']
y_test = test['angles']


##################################################
#
# Test prior models from 10 epochs on Udacity data
#
print("Model Testing:\n")
for i in range(epoch_nums):
    model_ith_arch = 'model' + str(i) + '.json'
    model_ith_parm = 'model' + str(i) + '.h5'
    if(os.path.exists(model_ith_arch)):
        with open(model_ith_arch, 'r') as jfile:
            model = model_from_json(json.load(jfile))
        model.compile(optimizer=adam, loss='mse')

    if(os.path.exists(model_ith_parm)):
        model.load_weights(model_ith_parm)

    loss = model.evaluate(X_test, y_test, verbose=1)
    print(loss)

