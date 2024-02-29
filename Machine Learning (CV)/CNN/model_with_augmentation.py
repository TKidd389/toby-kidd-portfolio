# -*- coding: utf-8 -*-
"""# Import Packages"""
from skimage.transform import resize

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import numpy as np

from tensorflow.keras import layers, models

import math

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import apply_affine_transform

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

"""# Hyperparameters"""

# Set seeds for Python, NumPy, and TensorFlow to ensure reproducibility
seed = 50
tf.random.set_seed(seed)
np.random.seed(seed)

keyword = "errorbars"

colab = False
include_errors = True
use_checkpoint = True

first_run = True
if first_run == True:
  use_checkpoint = False


percentTrain = 0.9
numParams = 1 #len(omegaParamArrayTrain[0])
numPredict = 10

batch_size = 64
EPOCHS = 200

checkpoint_loc = './checkpoints/checkpoint_' + str(keyword) + '.ckpt'

if colab == True:
  from google.colab import drive
  simulationFile = "shuffled_augmented_resized_Maps_Mtot_Nbody_SIMBA_LH_z=0.00.npy"
  parametersFile = "shuffled_augmented_resized_params_LH_Nbody_SIMBA.npy"
else:
  simulationFile = np.load("../../processed_data/256x256/Maps_Mtot_Nbody_SIMBA_LH_z=0.00.npy")
  parametersFile = np.load('../../processed_data/256x256/params_LH_Nbody_SIMBA.npy')

"""# Dataset

## Google Drive
"""

if colab == True:
  drive.mount('/content/drive')

"""## Create Datasets"""

def createDatasets(filename, percent):
  if colab == True:
    fgrids = '/content/drive/MyDrive/ColabNotebooks/' + filename
    grids  = np.load(fgrids)
  else:
    grids = simulationFile
  grids = resize(grids, (15000, 128, 128))
  lenValidation = math.ceil(len(grids)*percent)
  trainDataset = grids[:lenValidation]
  testDataset = grids[lenValidation:]
  return trainDataset, testDataset

dataset = createDatasets(simulationFile, percentTrain)
def preprocess_image(image):
  image = np.log(image)
  image /= np.max(image)
  return image
trainDataset = dataset[0]
testDataset = dataset[1]

trainDataset = preprocess_image(trainDataset)
testDataset = preprocess_image(testDataset)

print(len(trainDataset))
xImageRes = trainDataset.shape[-1]
yImageRes = trainDataset.shape[-2]
print(xImageRes, yImageRes)

"""## Create Parameter Labels"""

parameters = np.array(["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"])

def getParamArrays(filename, param, percent):
  if colab == True:
    filepath = '/content/drive/MyDrive/ColabNotebooks/' + filename
    paramArray  = np.load(filepath)
  else:
    paramArray = parametersFile
  lenValidation = math.ceil(len(paramArray)*percent)
  paramArrayTrain = paramArray[:lenValidation]
  paramArrayTest = paramArray[lenValidation:]
  return paramArrayTrain, paramArrayTest

omegaParamArray = getParamArrays(parametersFile, "Omega_m", percentTrain)

omegaParamArrayTrain = omegaParamArray[0]
omegaParamArrayTrain = np.repeat(omegaParamArrayTrain,15, axis=0)

omegaParamArrayTrain = np.array(omegaParamArrayTrain, dtype=float)

omegaParamArrayTest = omegaParamArray[1]
omegaParamArrayTest = np.repeat(omegaParamArrayTest,15, axis=0)

omegaParamArrayTest = np.array(omegaParamArrayTest, dtype=float)

print(omegaParamArrayTrain[126])

print(len(trainDataset))
print(len(omegaParamArrayTrain))
print(trainDataset.shape)
print(omegaParamArrayTrain.shape)

"""## Neural network inputs"""

#labels = omegaParamArrayTrain/np.max(omegaParamArrayTrain)

print(omegaParamArrayTrain)
print(omegaParamArrayTrain[:,5]/2)

labels = np.zeros((len(omegaParamArrayTrain), numParams))

labels[:,0] = np.ones(len(labels))
print(labels)

minmax = np.zeros((numParams,2))

for i in range(numParams):
  y_min = np.min(omegaParamArrayTrain[:,i])
  minmax[i][0] = y_min
  y_max = np.max(omegaParamArrayTrain[:,i])
  minmax[i][1] = y_max
  labels[:,i] = (omegaParamArrayTrain[:,i] - y_min) / (y_max - y_min)

"""# Network"""
"""## Data Processing"""

def custom_augmentation(x):
    angle = np.random.choice([0, 90, 180, 270])  # Choose from discrete angles
    x = apply_affine_transform(x, theta=angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    return x

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Assuming x_train has shape (13500, 128, 128)
trainDataset = np.expand_dims(trainDataset, axis=-1)  # Add channel dimension

# Assuming x_data and y_data are your original data
x_train, x_val, y_train, y_val = train_test_split(trainDataset, labels, test_size=0.2, random_state=20)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val)

# Create an instance of ImageDataGenerator with your custom augmentation
datagen = ImageDataGenerator(preprocessing_function=custom_augmentation)

# Create generators for training and validation data
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = datagen.flow(x_val, y_val, batch_size=batch_size)

"""## Model"""
"""
class RandomFlipAndRotateLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=False):
        if not training:
            return inputs
        
        # Random vertical flip
        outputs = tf.image.random_flip_up_down(inputs)

        # Random horizontal flip
        outputs = tf.image.random_flip_left_right(outputs)
        
        # Random rotation
        rotation_k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        outputs = tf.image.rot90(outputs, k=rotation_k)
        
        return outputs
"""
class RandomFlipAndRotateLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomFlipAndRotateLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Random horizontal flip
        flipped = tf.image.random_flip_left_right(inputs)
        # Random vertical flip
        flipped = tf.image.random_flip_up_down(flipped)
        # Random 90 degree rotations
        rotated = tf.image.rot90(flipped, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return rotated

def build_model(xImageRes, yImageRes, numParams):
    inputs = Input(shape=(xImageRes, yImageRes, 1))

    x = RandomFlipAndRotateLayer()(inputs)

    # Convolutional and pooling layers
    x = Conv2D(32, (3, 3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128, activation=tf.nn.leaky_relu)(x)
    x = Dropout(0.2)(x)

    # Output layers for mean and standard deviation
    mean_output = Dense(numParams, activation='sigmoid', name='mean')(x)
    stddev_output = Dense(numParams, activation='softplus', name='stddev')(x)

    return inputs, mean_output, stddev_output

def add_custom_loss(inputs, mean_output, stddev_output, numParams):
    y_true = Input(shape=(numParams,), name='y_true')
    def custom_loss(tensor):
        epsilon = 1e-6  # Small constant for numerical stability
        theta, mu, sigma = tensor

        diff = theta - mu
        sigma_squared = tf.square(sigma) + epsilon  # Ensure sigma is not zero

        # First term: Mean Squared Error between predictions and true values, normalized
        term1 = tf.reduce_mean(tf.square(diff))  # MSE without log to maintain scale and improve stability

        # Second term: Encourage sigma to capture the variance of the predictions
        term2 = tf.reduce_mean(tf.square(tf.square(diff) - sigma_squared))  # Similar to term1; no log for stability

        # Combine terms without exponential to avoid large values
        loss = term1 + term2

        return term1 #loss
    loss = Lambda(lambda x: custom_loss(x))([y_true, mean_output, stddev_output])
    model = Model(inputs=[inputs, y_true], outputs=[mean_output, stddev_output])
    model.add_loss(tf.reduce_mean(loss))
    return model

# Example usage
inputs, mean_output, stddev_output = build_model(xImageRes, yImageRes, numParams)
model = add_custom_loss(inputs, mean_output, stddev_output, numParams)

if use_checkpoint == True:
  # Create a callback that saves the model's weights for the lowest validation loss
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_loc,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True)
  model.load_weights(checkpoint_loc)


model.compile(optimizer='adam', metrics=['mae'])

# Prepare your data as [x_train, y_train] for training and [x_val, y_val] for validation
# Assuming x_train, y_train, x_val, y_val are already defined and preprocessed
if use_checkpoint ==True:
  history = model.fit([x_train, y_train], epochs=EPOCHS, callbacks=[early_stopping, checkpoint_callback], batch_size=32, validation_data=([x_val, y_val], None))
else:
  history = model.fit([x_train, y_train], epochs=EPOCHS, callbacks=[early_stopping], batch_size=32, validation_data=([x_val, y_val], None))

if first_run == True:
  model.save_weights(checkpoint_loc)

"""## Loss"""
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = np.arange(len(loss))#range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
if colab == False:
  plt.savefig('./images/loss_curve_'+str(keyword)+'.png')
else:
  plt.show()
