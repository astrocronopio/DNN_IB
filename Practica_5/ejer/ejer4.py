# Utilizando los datos y la arquitectura que considere oportuna 
# describir los distintos procesospara observar lo que la red 
# neuronal ha aprendido.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNet, mobilenet
from tensorflow.keras import losses, optimizers, metrics

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import shuffle

import tensorflow.keras as K
import tensorflow as tf