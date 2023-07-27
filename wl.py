# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Activation, Input, Concatenate
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import InputLayer, Reshape, Flatten, Dense, Dropout

from tensorflow.keras.constraints import Constraint
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
import tempfile
import random

from tqdm import tqdm

#Generate segmentation for image
import skimage
import skimage.io
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import copy
import sklearn.metrics
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries

import utils
from utils import *

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Load the datasets
x_train = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/x_train_full.npy')
y_train = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/y_train_full.npy')

x_test = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/x_test_full.npy')
y_test = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/y_test_full.npy')

x_val = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/x_val_full.npy')
y_val = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/y_val_full.npy')

# Create six subsets of the validation set for six-stage incremental learning
val_samples = len(x_val)

x_val_1 = x_val[:int(0.165 * val_samples),:,:,:]
y_val_1 = y_val[:int(0.165 * val_samples)]
print(x_val_1.shape)
print(np.unique(y_val_1))

x_val_2 = x_val[int(0.165 * val_samples):int(0.33 * val_samples),:,:,:]
y_val_2 = y_val[int(0.165 * val_samples):int(0.33 * val_samples)]
print(x_val_2.shape)
print(np.unique(y_val_2))

x_val_3 = x_val[int(0.33 * val_samples):int(0.495 * val_samples),:,:,:]
y_val_3 = y_val[int(0.33 * val_samples):int(0.495 * val_samples)]
print(x_val_3.shape)
print(np.unique(y_val_3))

x_val_4 = x_val[int(0.495 * val_samples):int(0.66 * val_samples),:,:,:]
y_val_4 = y_val[int(0.495 * val_samples):int(0.66 * val_samples)]
print(x_val_4.shape)
print(np.unique(y_val_4))

x_val_5 = x_val[int(0.66 * val_samples):int(0.825 * val_samples),:,:,:]
y_val_5 = y_val[int(0.66 * val_samples):int(0.825 * val_samples)]
print(x_val_5.shape)
print(np.unique(y_val_5))

x_val_6 = x_val[int(0.825 * val_samples):int(1 * val_samples),:,:,:]
y_val_6 = y_val[int(0.825 * val_samples):int(1 * val_samples)]
print(x_val_6.shape)
print(np.unique(y_val_6))

# set training hyperparameters
epochs = 10
lambda_ = 1
lr = 0.001
num_sample = int(0.05 * x_train.shape[0])
opt = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = weighted_scce_loss

# set the folder path to save the model files
dir = '/home/audio_ml.work/data/audio/speech/speech_commands_arrays/Trained_Models/'

# start baseline training with regular sample weights
sample_weight = np.ones((x_train.shape[0],))
train_A = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight)).shuffle(x_train.shape[0]).batch(512)
test_A = (x_val_1, y_val_1)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
conv = conv_model(input_shape=input_shape, num_classes=35)

model = conv.get_compiled_model(opt, loss_fn, ['sparse_categorical_accuracy'])

trn = Train(opt, loss_fn)
acc_prior_A = trn.train(model, epochs, train_A, test_tasks=[test_A], model_save_fname='session_0.h5')

# generate the fisher samples from EWC
fisher_samples = gen_fisher_samples(x_train, y_train, model)

data = [x_val_1, x_val_2, x_val_3, x_val_4, x_val_5, x_val_6]
labels = [y_val_1, y_val_2, y_val_3, y_val_4, y_val_5, y_val_6]
wt_file = ['session_0', 'session_1_wl', 'session_2_wl', 'session_3_wl', 'session_4_wl', 'session_5_wl', 'session_6_wl'] 

# Perform weighted-loss based EWC in six sessions
for f in range(6):
    model = tf.keras.models.load_model(dir+wt_file[f], custom_objects = {"weighted_scce_loss": weighted_scce_loss})
    new_train, new_labels, sample_weights = augment_incorrect_samples(x_train, y_train, [data[f]], [labels[f]], model)

    if f==0:
        val_data = [data[f]]
        val_labels = [labels[f]]
    elif f==1:
        val_data = [data[f-1], data[f]]
        val_labels = [labels[f-1], labels[f]]
    elif f==2:
        val_data = [data[f-2], data[f-1], data[f]]
        val_labels = [labels[f-2], labels[f-1], labels[f]]
    elif f==3:
        val_data = [data[f-3], data[f-2], data[f-1], data[f]]
        val_labels = [labels[f-3], labels[f-2], labels[f-1], labels[f]]  
    elif f==4:
        val_data = [data[f-4], data[f-3], data[f-2], data[f-1], data[f]]
        val_labels = [labels[f-4], labels[f-3], labels[f-2], labels[f-1], labels[f]]  
    elif f==5:
        val_data = [data[f-5], data[f-4], data[f-3], data[f-2], data[f-1], data[f]]
        val_labels = [labels[f-5], labels[f-4], labels[f-3], labels[f-2], labels[f-1], labels[f]]  

    i = 1
    for d, l in zip(val_data, val_labels):
        y_pred = np.argmax(model.predict(d), axis=1)
        wrong_pred = np.where(y_pred != l)
        print('Number of incorrect predictions in validation set '+repr(i)+': '+repr(len(wrong_pred[0])))
        i = i+1
    
    # ewc = EWC(model, fisher_samples, num_sample=num_sample)
    # f_matrix = ewc.get_fisher()

    train = tf.data.Dataset.from_tensor_slices((new_train, new_labels, sample_weights)).shuffle(new_train.shape[0]).batch(512)

    if f==5:
        test = (x_test, y_test)
    else:
        test = (data[f+1], labels[f+1])

    # prior_weights = model.get_weights()
    print('\n [INFO] Starting Training Session '+repr(f+1))

    trn = Train(opt, loss_fn, prior_weights=None, lambda_=None)
    acc = trn.train(model, 
                    epochs, 
                    train, 
                    fisher_matrix=None, 
                    test_tasks=[test],
                    model_save_fname = dir+wt_file[f+1]
                    )
    print('[INFO] ACC: {}'.format(acc))

    # if f != 5:
    #     fisher_samples = gen_fisher_samples(new_train, new_labels, model)
print('\n')

model = tf.keras.models.load_model(dir+'session_6_ewc', custom_objects = {"weighted_scce_loss": weighted_scce_loss})

i = 1
for d, l in zip(data, labels):
    y_pred = np.argmax(model.predict(d), axis=1)
    wrong_pred = np.where(y_pred != l)
    print('Number of incorrect predictions in validation set '+repr(i)+': '+repr(len(wrong_pred[0])))
    i = i+1
print('\n')

# Print out test set performance at each incremental learning stage
s = 1
for i in wt_file:
    # model = conv.get_compiled_model(opt, loss_fn, ['sparse_categorical_accuracy'])
    model = tf.keras.models.load_model(dir+i, custom_objects = {"weighted_scce_loss": weighted_scce_loss})
    
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = y_test
    
    test_acc = sum(y_pred == y_true) / len(y_true)
    print('Test set accuracy after session '+repr(s)+': '+repr(round(100*test_acc,2)))
    s = s + 1