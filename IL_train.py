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

# Generate segmentation for image
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
import argparse

def load_data():
    # Load the datasets
    x_train = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/x_train_full.npy')
    y_train = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/y_train_full.npy')

    x_test = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/x_test_full.npy')
    y_test = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/y_test_full.npy')

    x_val = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/x_val_full.npy')
    y_val = np.load('/home/audio_ml.work/data/audio/speech/speech_commands_arrays/new_split/y_val_full.npy')

    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

    return x_train, y_train, x_test, y_test, x_val, y_val, input_shape


def setup_training_hyperparameters():
    # Set training hyperparameters
    epochs = 15
    lambda_ = 1
    lr = 0.002
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = weighted_scce_loss 
    batch_size = 512

    return epochs, lambda_, lr, opt, loss_fn, batch_size

class DataPartitioner:
    def __init__(self, x_val, y_val, num_partitions, mode):
        self.x_val = x_val
        self.y_val = y_val
        self.num_partitions = num_partitions
        self.val_samples = len(x_val)
        self.data = []
        self.labels = []
        self.wt_file = []
        self.mode = mode

    def get_partition(self):
        for i in range(0, self.num_partitions):
            idx = int(1.0 / self.num_partitions * self.val_samples) #set number of samples for each partition
            start = idx * i                                         #set the starting index for a partition
            end = (idx) * (i+1)                                     #set the concluding index for a partition
            X = self.x_val[start:end,:,:,:]                         #store partitioned data array into a variable
            y = self.y_val[start:end]                               #store partitioned label array into a variable
            self.data.append(X)                                     #populate the data list with the data arrays
            self.labels.append(y)                                   #populate the label list with the label arrays
        for j in range(0,self.num_partitions+1):    
            wt = 'session_'+str(j)+'_'+str(self.mode)  
            self.wt_file.append(wt)                                 #store session names in a list
        return self.data, self.labels, self.wt_file
    
class IncrementalLearning:
    def __init__(self, x_train, y_train, partitioner, model, num_sample, 
                 train_opts, save_dir, wt_file, data, labels, mode, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.partitioner = partitioner
        self.model = model
        # self.fisher_samples = fisher_samples                      #making fisher samples exclusive to '_train_partitions' as we don't need fisher samples for other two modes (traditional and weighted loss)
        self.num_sample = num_sample
        self.train_opts = train_opts
        self.save_dir = save_dir
        self.wt_file = wt_file
        self.data = data
        self.labels = labels
        self.mode = mode
        self.x_test = x_test
        self.y_test = y_test

    def _print_wrong_predictions(self, val_data, val_labels, model):
        i = 1
        for d, l in zip(val_data, val_labels):
            y_pred = np.argmax(model.predict(d), axis=1)
            wrong_pred = np.where(y_pred != l)
            print('Number of incorrect predictions in validation set ' + repr(i) + ': ' + repr(len(wrong_pred[0])))
            i = i + 1
        print('\n')

    def _print_test_performances(self):
        for i in range(self.partitioner.num_partitions+1):
            model = tf.keras.models.load_model(self.save_dir + self.wt_file[i],
                                               custom_objects={"weighted_scce_loss": weighted_scce_loss})
            y_pred = np.argmax(model.predict(self.x_test), axis=1)
            y_true = self.y_test
            test_acc = sum(y_pred == y_true) / len(y_true)
            print('Test set accuracy after session ' + repr(i) + ': ' + repr(round(100 * test_acc, 2)))    

    def _base_train(self):
            # start baseline training with regular sample weights
            sample_weight = np.ones((self.x_train.shape[0],))
            train_base = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train, 
                                                            sample_weight)).shuffle(self.x_train.shape[0]).batch(
                                                                                    self.train_opts.get('batch_size'))
            test_base = (self.x_test, self.y_test) #test on the test set
            trn = Train(self.train_opts.get('optimizer'), self.train_opts.get('loss_fn'))
            acc_base = trn.train(self.model, self.train_opts.get('epochs'), 
                                 train_base, test_tasks=[test_base], 
                                 model_save_fname=self.save_dir+'session_0_'+str(self.mode))
            return self.model
            
    def _train_partitions(self, model, fisher_samples=None):
        for i in range(self.partitioner.num_partitions):
            model = tf.keras.models.load_model(self.save_dir + self.wt_file[i],
                                               custom_objects={"weighted_scce_loss": weighted_scce_loss})  
            new_train, new_labels, sample_weights = augment_incorrect_samples(self.x_train, self.y_train, 
                                                                              [self.data[i]], [self.labels[i]], 
                                                                              model, self.mode)
            if i==0: val_data = [self.data[i]]; val_labels=[self.labels[i]]
            else: 
                val_data = []; val_labels = []
                val_data.extend(self.data[0:i+1]); val_labels.extend(self.labels[0:i+1])
            
            self._print_wrong_predictions(val_data, val_labels, model)
            if self.mode == 'ewc':
                ewc = EWC(model, fisher_samples, num_sample=self.num_sample)
                f_matrix = ewc.get_fisher()
            else: f_matrix = None

            train = tf.data.Dataset.from_tensor_slices((new_train, new_labels, 
                                                        sample_weights)).shuffle(new_train.shape[0]).batch(
                                                                                    self.train_opts.get('batch_size'))
            prior_weights = model.get_weights()
            print('\n [INFO] Starting Training Session '+repr(i+1))   

            trn = Train(self.train_opts.get('optimizer'), self.train_opts.get('loss_fn'), 
                        prior_weights=prior_weights, lambda_=self.train_opts.get('lambda_'))
            acc = trn.train(model, self.train_opts.get('epochs'), train, 
                            fisher_matrix=f_matrix, test_tasks=[(self.x_test, self.y_test)],
                            model_save_fname = self.save_dir+self.wt_file[i+1])
            print('[INFO] TEST ACC: {}'.format(acc))   

            if i != self.partitioner.num_partitions - 1:
                fisher_samples = gen_fisher_samples(new_train, new_labels, model)    
          

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, help='Provide the number of incremental sessions')
    parser.add_argument("--mode", type=str, help='Provide method to use for IL (options are trad, wl or ewc)')
    parser.add_argument("--folder", type=str, help='Provide the folder name to save the trained models')
    parser.add_argument("--seed",type=int, help='Provide seed value for reproducibility')
    args = parser.parse_args()

    # Set the seed value for experiment reproducibility.
    seed = args.seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_partitions = args.sessions
    mode = args.mode
    folder = args.folder

    x_train, y_train, x_test, y_test, x_val, y_val, input_shape = load_data()

    epochs, lambda_, lr, opt, loss_fn, batch_size = setup_training_hyperparameters()

    num_sample = int(0.05 * x_train.shape[0])

    # Set the folder path to save the model files
    output_dir = '/home/audio_ml.work/data/audio/speech/speech_commands_arrays/Trained_Models/EUSIPCO/'+str(mode)+'/Models/'+str(folder)+'/'

    partitioner = DataPartitioner(x_val=x_val, y_val=y_val, num_partitions=num_partitions, mode=mode)
    data, labels, wt_file = partitioner.get_partition()
    conv = conv_model(input_shape=input_shape, num_classes=35)
    model = conv.get_compiled_model(opt, loss_fn, ['sparse_categorical_accuracy'])

    trainer = IncrementalLearning(x_train, y_train, partitioner, model, num_sample,
                                  {'optimizer': opt, 'loss_fn': loss_fn, 'epochs': epochs, 
                                   'lambda_': lambda_, 'batch_size': batch_size},
                                  output_dir, wt_file, data, labels, mode, x_test, y_test)
    model = trainer._base_train()

    # # generate the fisher samples from EWC
    if mode == 'ewc': fisher_samples = gen_fisher_samples(x_train, y_train, model)    #generate fisher samples after baseline training. It will be forwarded to IL training if the mode is 'ewc'
    else: fisher_samples = None

    trainer._train_partitions(model, fisher_samples)

    model = tf.keras.models.load_model(output_dir+wt_file[-1],          #load the final session weights 
                                       custom_objects = {"weighted_scce_loss": weighted_scce_loss})
    print('\n')
    trainer._print_wrong_predictions(data, labels, model)
    print('\n')
    trainer._print_test_performances()

# Call the main function
if __name__ == "__main__":
    main()