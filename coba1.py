#training data processing
import numpy as np
import h5py
from keras.datasets import mnist
from keras.utils import to_categorical
import argparse
import os
import os
import glob
import numpy 
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import cv2                 
from random import shuffle
from tqdm import tqdm
import skimage.data 

import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _mean_squared_error(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def _mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _identity(x):
    return x

class ELM(object):
    def __init__(
        self, n_input_nodes, n_hidden_nodes, n_output_nodes,
        activation='sigmoid', loss='mean_squared_error', name=None,
        beta_init=None, alpha_init=None, bias_init=None):

        self.name = name
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes

        # initialize weights and a bias
        if isinstance(beta_init, np.ndarray):
            if beta_init.shape != (self.__n_hidden_nodes, self.__n_output_nodes):
                raise ValueError(
                    'the shape of beta_init is expected to be (%d,%d).' % (self.__n_hidden_nodes, self.__n_output_nodes)
                )
            self.__beta = beta_init
        else:
            self.__beta = np.random.uniform(-1.,1.,size=(self.__n_hidden_nodes, self.__n_output_nodes))
        if isinstance(alpha_init, np.ndarray):
            if alpha_init.shape != (self.__n_input_nodes, self.__n_hidden_nodes):
                raise ValueError(
                    'the shape of alpha_init is expected to be (%d,%d).' % (self.__n_hidden_nodes, self.__n_output_nodes)
                )
            self.__alpha = alpha_init
        else:
            self.__alpha = np.random.uniform(-1.,1.,size=(self.__n_input_nodes, self.__n_hidden_nodes))
        if isinstance(bias_init, np.ndarray):
            if bias_init.shape != (self.__n_hidden_nodes,):
                raise ValueError(
                    'the shape of bias_init is expected to be (%d,).' % (self.__n_hidden_nodes,)
                )
            self.__bias = bias_init
        else:
            self.__bias = np.zeros(shape=(self.__n_hidden_nodes,))

        # set an activation function
        self.__activation = self.__get_activation_function(activation)

        # set a loss function
        self.__loss = self.__get_loss_function(loss)

    def __call__(self, x):
        x = np.array(x)
        h = self.__activation(x.dot(self.__alpha) + self.__bias)
        return h.dot(self.__beta)

    def predict(self, x):
        return list(self(x))

    def evaluate(self, x, t, metrics=['loss']):
        y_pred = self.predict(x)
        y_true = t
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        y_true_argmax = np.argmax(y_true, axis=-1)
        ret = []
    def __get_activation_name(self, activation):
        for m in metrics:
            if m == 'loss':
                loss = self.__loss(y_true, y_pred)
                ret.append(loss)
            elif m == 'accuracy':
                acc = np.sum(y_pred_argmax == y_true_argmax) / len(t)
                ret.append(acc)
            elif m == 'uar':
                num_classes = len(t[0])
                uar = []
                for i in range(num_classes):
                    tp = np.sum((y_pred_argmax == i) & (y_true_argmax == i))
                    tp_fn = np.sum(y_true_argmax == i)
                    uar.append(tp / tp_fn)
                uar = np.mean(uar)
                ret.append(uar)
            else:
                raise ValueError(
                    'an unknown evaluation indicator \'%s\'.' % m
                )
        if len(ret) == 1:
            ret = ret[0]
        elif len(ret) == 0:
            ret = None
        return ret


    def fit(self, x, t):
        x = np.array(x)
        H = self.__activation(x.dot(self.__alpha) + self.__bias)

        # compute a pseudoinverse of H
        H_pinv = np.linalg.pinv(H)

        # update beta
        self.__beta = H_pinv.dot(t)

    def save(self, filepath):
        with h5py.File(filepath, 'w') as f:
            arc = f.create_dataset('architecture', data=np.array([self.__n_input_nodes, self.__n_hidden_nodes, self.__n_output_nodes]))
            arc.attrs['activation'] = self.__get_activation_name(self.__activation).encode('utf-8')
            arc.attrs['loss'] = self.__get_loss_name(self.__loss).encode('utf-8')
            arc.attrs['name'] = self.name.encode('utf-8')
            f.create_group('weights')
            f.create_dataset('weights/alpha', data=self.__alpha)
            f.create_dataset('weights/beta', data=self.__beta)
            f.create_dataset('weights/bias', data=self.__bias)

    def __get_activation_function(self, name):
        if name == 'sigmoid':
            return _sigmoid
        elif name == 'identity':
            return _identity
        else:
            raise ValueError(
                'an unknown activation function \'%s\'.' % name
            )

    def __get_activation_name(self, activation):
        if activation == _sigmoid:
            return 'sigmoid'
        elif activation == _identity:
            return 'identity'

    def __get_loss_function(self, name):
        if name == 'mean_squared_error':
            return _mean_squared_error
        elif name == 'mean_absolute_error':
            return _mean_absolute_error
        else:
            raise ValueError(
                'an unknown loss function \'%s\'.' % name
            )

    def __get_loss_name(self, loss):
        if loss == _mean_squared_error:
            return 'mean_squared_error'
        elif loss == _mean_absolute_error:
            return 'mean_absolute_error'
    
    @property
    def weights(self):
        return {
            'alpha': self.__alpha,
            'beta': self.__beta,
            'bias': self.__bias,
        }

    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    @property
    def output_shape(self):
        return (self.__n_output_nodes,)

    @property
    def n_input_nodes(self):
        return self.__n_input_nodes

    @property
    def n_hidden_nodes(self):
        return self.__n_hidden_nodes

    @property
    def n_output_nodes(self):
        return self.__n_output_nodes

    @property
    def activation(self):
        return self.__get_activation_name(self.__activation)

    @property
    def loss(self):
        return self.__get_loss_name(self.__loss)

def load_model(filepath):
    with h5py.File(filepath, 'r') as f:32
        alpha_init = f['weights/alpha'][...]
        beta_init = f['weights/beta'][...]
        bias_init = f['weights/bias'][...]
        arc = f['architecture']
        n_input_nodes = arc[0]
        n_hidden_nodes = arc[1]
        n_output_nodes = arc[2]
        activation = arc.attrs['activation'].decode('utf-8')
        loss = arc.attrs['loss'].decode('utf-8')
        name = arc.attrs['name'].decode('utf-8')
        model = ELM(
            n_input_nodes=n_input_nodes,
            n_hidden_nodes=n_hidden_nodes,
            n_output_nodes=n_output_nodes,
            activation=activation,
            loss=loss,
            alpha_init=alpha_init,
            beta_init=beta_init,
            bias_init=bias_init,
            name=name,
        )
    return model

def softmax(x):
    x = np.array(x)
    c = np.max(x, axis=-1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=-1)
    return upper / lower


model = load_model("model.h5")
img = cv2.imread("101.png",cv2.IMREAD_GRAYSCALE)
plt.subplot(121),plt.imshow(img, cmap='gray')
plt.show()
img = cv2.resize(img, (50,50)) 
test_img = numpy.array(img)
edges = cv2.Canny(img, 100, 200)
test = numpy.array(img)
test = test.flatten()
result_pred = softmax(model.predict(test))
class_pred = np.argmax(result_pred)
prob_pred = result_pred[class_pred]
print(result_pred)
print(class_pred)
print(prob_pred)