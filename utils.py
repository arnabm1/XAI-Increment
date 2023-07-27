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

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def evaluate(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = y_test
    
    test_acc = sum(y_pred == y_true) / len(y_true)
    # print('Test set accuracy after session '+repr(s)+': '+repr(round(100*test_acc,2)))
    return round(100*test_acc,2)

def weighted_scce_loss(y_true, y_pred, sample_weight):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = scce(y_true, y_pred, sample_weight=sample_weight)
    return loss

class Train:
    
    def __init__(self, optimizer, loss_fn, prior_weights=None, lambda_=0.1):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prior_weights = prior_weights
        self.lambda_ = lambda_
        
    def train(self, model, epochs, train_task, fisher_matrix=None, test_tasks=None, model_save_fname=None):
        # empty list to collect per epoch test acc of each task
        updated_result = 0
        if test_tasks:
            test_acc = [[] for _ in test_tasks]
        else: 
            test_acc = None
        for epoch in tqdm(range(epochs)):
            for batch in train_task:
                X, y, wt = batch
                with tf.GradientTape() as tape:
                    pred = model(X)
                    loss = self.loss_fn(y, pred, wt)
                    # if to execute training with EWC
                    if fisher_matrix is not None:
                        loss += self.compute_penalty_loss(model, fisher_matrix)
                grads = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # evaluate with the test set of task after each epoch
            if test_acc:
                for i in range(len(test_tasks)):
                    result = evaluate(model, test_tasks[i][0], test_tasks[i][1])
                    if result > updated_result:
                        print('Accuracy increased from '+repr(updated_result)+'% to '+repr(result)+'%,saving model weights')
                        updated_result = result
                        model.save(model_save_fname, save_format='tf')
                    test_acc[i].append(result)        
        return max(test_acc[0])

    def compute_penalty_loss(self, model, fisher_matrix):
        penalty = 0.
        for u, v, w in zip(fisher_matrix, model.weights, self.prior_weights):
            penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
        return 0.5 * self.lambda_ * penalty
    

class EWC:
    
    def __init__(self, prior_model, data_samples, num_sample=30):
        self.prior_model = prior_model
        self.prior_weights = prior_model.weights
        self.num_sample = num_sample
        self.data_samples = data_samples
        self.fisher_matrix = self.compute_fisher()
        
    def compute_fisher(self):
        weights = self.prior_weights
        fisher_accum = np.array([np.zeros(layer.numpy().shape) for layer in weights], 
                           dtype=object
                          )
        for j in range(self.num_sample):
            idx = np.random.randint(self.data_samples.shape[0])
            with tf.GradientTape() as tape:
                logits = tf.nn.log_softmax(self.prior_model(np.array([self.data_samples[idx]])))
            grads = tape.gradient(logits, weights)
            for m in range(len(weights)):
                fisher_accum[m] += np.square(grads[m])
        fisher_accum /= self.num_sample
        return fisher_accum
    
    def get_fisher(self):
        return self.fisher_matrix
    

class conv_model:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_conv()
        
    def create_conv(self):
        input       = Input(shape=self.input_shape)

        rz          = tf.keras.layers.Resizing(64,64)(input)
        conv        = Conv2D(filters=8, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(rz)
        conv_act    = activations.relu(conv)
        conv        = Conv2D(filters=8, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)    

        conv        = Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)
        conv        = Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)

        conv        = Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)
        conv        = Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)
        conv        = Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        pool        = MaxPooling2D(pool_size=2, strides=2, padding ='same')(conv)
        conv_act    = activations.relu(pool)

        conv        = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)
        conv        = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        conv_act    = activations.relu(conv)
        conv        = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=(1,1))(conv_act)
        pool        = MaxPooling2D(pool_size=2, strides=2, padding ='same')(conv)
        conv_act    = activations.relu(pool)

        flat        = Flatten()(conv_act)

        dense       = Dense(1000, kernel_initializer='he_normal')(flat)
        act         = activations.relu(dense)
        dense       = Dense(self.num_classes, activation='softmax')(act)
        model       = models.Model(inputs=input, outputs=dense)
        return model
    
    
    def get_uncompiled_model(self):
        return self.model
    
    def get_compiled_model(self, optimizer, loss_fn, metrics ):
        compiled_model = self.model
        compiled_model.compile(optimizer, loss_fn, metrics)
        return compiled_model

def lime_distances(new_samples, new_samples_labels, model):
    
    def perturb_image(img,perturbation,segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image*mask[:,:,np.newaxis]
        return perturbed_image

    lime = []

    for specs, labels in zip(new_samples, new_samples_labels):
        images = np.asarray(specs).astype('double')
        images = np.reshape(images, (images.shape[0],images.shape[1],images.shape[2]))

        superpixels = skimage.segmentation.slic(images, n_segments=10, compactness=10, sigma=1, start_label=1)
        num_superpixels = np.unique(superpixels).shape[0]

        num_perturb = 15
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
        predictions = []
        for pert in perturbations:
            perturbed_img = perturb_image(images,pert,superpixels)
            pred = model.predict(perturbed_img[np.newaxis,:,:,:])
            predictions.append(pred)
        predictions = np.array(predictions)

        original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
        distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
        #Transform distances to a value between 0 an 1 (weights) using a kernel function
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
        preds = model.predict(images[np.newaxis,:,:,:])
        top_pred_classes = preds[0].argsort()[-5:][::-1] # Save ids of top 5 classes

        
        class_to_explain = top_pred_classes[0]
        simpler_model = LinearRegression()
        correct_ = LinearRegression()
        simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
        correct_.fit(X=perturbations, y=predictions[:,:,labels], sample_weight=weights)
        coeff = simpler_model.coef_[0]
        correct_coeff = correct_.coef_[0]

        coeff_shifted = coeff + max(abs(coeff))
        correct_coeff_shifted = correct_coeff + max(abs(correct_coeff))

        norm_coeff = coeff_shifted/max(coeff_shifted)
        norm_correct_coeff = correct_coeff_shifted/max(correct_coeff_shifted)

        # diff = np.sum(np.square(coeff - correct_coeff))
        diff = np.sum(np.square(norm_coeff - norm_correct_coeff))
        lime.append(diff)
    return np.array(lime)


def augment_incorrect_samples(x_train, y_train, val_data, val_labels, model, mode):
    w_samples = []
    w_labels = []

    for data, labels in zip(val_data, val_labels):
        y_pred = np.argmax(model.predict(data), axis=1)
        wrong_pred = np.where(y_pred != labels)

        for indexes in wrong_pred[0]:
            w_samples.append(data[indexes,:,:,:])
            w_labels.append(labels[indexes])

    w_samples = np.array(w_samples)
    w_labels = np.array(w_labels)

    wt_train = np.ones((x_train.shape[0],)) - 0.5

    if mode == 'trad': wt_w = np.ones((w_samples.shape[0],)) - 0.5
    else: wt_w = 1 + lime_distances(w_samples, w_labels, model); wt_train = np.ones((x_train.shape[0],)) 
    
    new_train = np.concatenate([x_train, w_samples])
    new_labels = np.concatenate([y_train, w_labels])
    sample_weights = np.concatenate([wt_train, wt_w])
    print('Updated number of training samples: '+repr(new_train.shape[0]))

    return new_train, new_labels, sample_weights


def gen_fisher_samples(x_train, y_train, model):
    fisher_samples = []
    for data, labels in zip([x_train], [y_train]):
        y_pred = np.argmax(model.predict(data), axis=1)
        correct_pred = np.where(y_pred == labels)

        for indexes in correct_pred[0]:
            fisher_samples.append(data[indexes,:,:,:])
        # print('Number of correct predictions in training set '+': '+repr(len(correct_pred[0])))
    fisher_samples = np.array(fisher_samples)
    return fisher_samples

