"""# Import Packages"""
from skimage.transform import resize

#I want to change this so that all variables sit in a master document and get pulled out of there, difficult with colab, easier on machine

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

percentTrain = 0.9
numParams = 1 #len(omegaParamArrayTrain[0])
numPredict = 1500
numDataPoints = 50

batch_size = 64
EPOCHS = 1

xImageRes, yImageRes = 128, 128

checkpoint_loc = './checkpoints/checkpoint_' + str(keyword) + '.ckpt'

if colab == True:
  from google.colab import drive
  simulationFile = "shuffled_augmented_resized_Maps_Mtot_Nbody_SIMBA_LH_z=0.00.npy"
  parametersFile = "shuffled_augmented_resized_params_LH_Nbody_SIMBA.npy"
else:
  simulationFile = np.load("../../processed_data/256x256/Maps_Mtot_Nbody_SIMBA_LH_z=0.00.npy")
  parametersFile = np.load('../../processed_data/256x256/params_LH_Nbody_SIMBA.npy')
"""
else:
  simulationFile = np.load("../../processed_data/shuffled_augmented_resized_Maps_Mtot_Nbody_SIMBA_LH_z=0.00.npy")
  parametersFile = np.load('../../processed_data/shuffled_augmented_resized_params_LH_Nbody_SIMBA.npy')
"""
"""# Dataset

## Google Drive
"""

if colab == True:
  drive.mount('/content/drive')

"""## Create Datasets"""

def createDatasets(filename, percent):
  if colab == True:
    fgrids = '/content/drive/MyDrive/ColabNotebooks/' + filename
    grids  = np.load(fgrids)#, mmap_mode='r')
  else:
    grids = simulationFile
    print("length of grids")
    print(grids.shape)
  grids = resize(grids, (15000, 128, 128))
  lenValidation = math.ceil(len(grids)*percent)
  trainDataset = grids[:lenValidation]
  testDataset = grids[lenValidation:]
  return trainDataset, testDataset

dataset = createDatasets(simulationFile, percentTrain)
trainDataset = dataset[0]
testDataset = dataset[1]

print(len(trainDataset))
xImageRes = trainDataset.shape[-1]
yImageRes = trainDataset.shape[-2]
print(xImageRes, yImageRes)

def preprocess_image(image):
  image = np.log(image)
  image /= np.max(image)
  return image

trainDataset = preprocess_image(trainDataset)
testDataset = preprocess_image(testDataset)
print("Lengths:")
print(len(trainDataset))
print(len(testDataset))

"""## Create Parameter Labels"""

parameters = np.array(["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"])

def getParamArrays(filename, param, percent):
  if colab == True:
    filepath = '/content/drive/MyDrive/ColabNotebooks/' + filename
    paramArray  = np.load(filepath)#, mmap_mode='r')
  else:
    paramArray = parametersFile
  lenValidation = math.ceil(len(paramArray)*percent)
  paramArrayTrain = paramArray[:lenValidation]
  paramArrayTest = paramArray[lenValidation:]
  return paramArrayTrain, paramArrayTest

omegaParamArray = getParamArrays(parametersFile, "Omega_m", percentTrain)

omegaParamArrayTrain = omegaParamArray[0]
omegaParamArrayTrain = np.repeat(omegaParamArrayTrain,15, axis=0)
#if omegaParamArrayTrain[0] == omegaParamArrayTrain[1]:
#  print("success asdf")
print("1 array: " + str(omegaParamArrayTrain[0]) + " 2 array: " + str(omegaParamArrayTrain[2]))
#length = len(omegaParamArrayTrain)
#omegaParamArrayTrain = np.reshape(omegaParamArrayTrain, (length // 6, 6))
#omegaParamArrayTrain = np.tile(omegaParamArrayTrain, (2, 1))

omegaParamArrayTrain = np.array(omegaParamArrayTrain, dtype=float)

omegaParamArrayTest = omegaParamArray[1]
omegaParamArrayTest = np.repeat(omegaParamArrayTest,15, axis=0)
#length = len(omegaParamArrayTest)
#omegaParamArrayTest = np.reshape(omegaParamArrayTest, (length // 6, 6))
#omegaParamArrayTest = np.tile(omegaParamArrayTest, (2, 1))

omegaParamArrayTest = np.array(omegaParamArrayTest, dtype=float)

"""
omegaParamArray = getParamArrays(parametersFile, "Omega_m", percentTrain)

omegaParamArrayTrain = omegaParamArray[0]
#omegaParamArrayTrain = np.repeat(omegaParamArrayTrain,15)

omegaParamArrayTrain = np.array(omegaParamArrayTrain, dtype=float)

omegaParamArrayTest = omegaParamArray[1]
#omegaParamArrayTest = np.repeat(omegaParamArrayTest,15)

omegaParamArrayTest = np.array(omegaParamArrayTest, dtype=float)

print(omegaParamArrayTrain[126])

print(len(trainDataset))
print(len(omegaParamArrayTrain))
print(trainDataset.shape)
print(omegaParamArrayTrain.shape)
"""
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

"""# Prediction"""

"""## Prediction Model"""
"""
def build_prediction_model(xImageRes, yImageRes, numParams):
    inputs = Input(shape=(xImageRes, yImageRes, 1))

    # Convolutional and pooling layers
    x = Conv2D(32, (3, 3), activation=tf.nn.leaky_relu)(inputs)
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

    model = Model(inputs=inputs, outputs=[mean_output, stddev_output])

    return model
"""
def build_prediction_model(xImageRes, yImageRes, numParams):
    inputs = Input(shape=(xImageRes, yImageRes, 1))

    x = Conv2D(32, (3, 3), activation=tf.nn.leaky_relu)(inputs)
#    x = Conv2D(32, (3, 3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation=tf.nn.leaky_relu)(x)
#    x = Conv2D(64, (3, 3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation=tf.nn.leaky_relu)(x)
#    x = Conv2D(128, (3, 3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation=tf.nn.leaky_relu)(x)
#    x = GlobalAveragePooling2D()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation=tf.nn.leaky_relu)(x)
    x = Dropout(0.3)(x)

    mean_output = Dense(numParams, activation='sigmoid', name='mean')(x)
    stddev_output = Dense(numParams, activation='softplus', name='stddev')(x)

    model = Model(inputs=inputs, outputs=[mean_output, stddev_output])
    return model


prediction_model = build_prediction_model(xImageRes, yImageRes, numParams)

# Assuming "model" was used for training
# Transfer weights to prediction_model if they are indeed separate models
#prediction_model.set_weights(model.get_weights())
prediction_model.load_weights(checkpoint_loc)

"""## Difference"""

trainDataset[0].shape

if include_errors == True:
  y_pred_mean, y_pred_stddev = prediction_model.predict(testDataset[0:numPredict])
  predictions = np.zeros((len(y_pred_mean),numParams))
  stddev_predictions = np.zeros((len(y_pred_stddev),numParams))
  for i in range(numParams):
    predictions[:,i] = y_pred_mean[:,i]*(minmax[i][1]-minmax[i][0])+minmax[i][0]
    stddev_predictions[:,i] = y_pred_stddev[:,i]*(minmax[i][1]-minmax[i][0])+minmax[i][0]

  paramArrayPredict = omegaParamArrayTrain[0:numPredict][:,:numParams]#omegaParamArrayTest

  # Calculate the absolute differences between predicted and actual values
  differences = np.abs(predictions - paramArrayPredict)

  # Calculate the average difference
  average_difference = np.mean(differences, axis=0)

  print(f"Average Difference: {average_difference}")

  # Calculate the percentage differences between predicted and actual values
  percentage_differences = 100 * np.abs(predictions - paramArrayPredict) / np.abs(paramArrayPredict)

  # Calculate the average percentage difference
  average_percentage_difference = np.mean(percentage_differences, axis=0)

  print(f"Average Percentage Difference: {average_percentage_difference}%")
  print(f"Average Error: {np.mean(stddev_predictions)*100}%")

if include_errors == False:
  # Make predictions on the test dataset
  predicted = model.predict(testDataset[0:numPredict])

  predictions = np.zeros((len(predicted),numParams))
  for i in range(numParams):
    predictions[:,i] = predicted[:,i]*(minmax[i][1]-minmax[i][0])+minmax[i][0]
  #predictions = predicted*(y_max - y_min) + y_min#predicted*np.max(omegaParamArrayTrain)

  paramArrayPredict = omegaParamArrayTrain[0:numPredict][:,:numParams]#omegaParamArrayTest

  # Calculate the absolute differences between predicted and actual values
  differences = np.abs(predictions - paramArrayPredict)

  # Calculate the average difference
  average_difference = np.mean(differences, axis=0)

  print(f"Average Difference: {average_difference}")

  # Calculate the percentage differences between predicted and actual values
  percentage_differences = 100 * np.abs(predictions - paramArrayPredict) / np.abs(paramArrayPredict)

  # Calculate the average percentage difference
  average_percentage_difference = np.mean(percentage_differences, axis=0)

  print(f"Average Percentage Difference: {average_percentage_difference}%")


if include_errors == True:
  y_pred_mean, y_pred_stddev = prediction_model.predict(testDataset[0:numPredict])
  predictions = np.zeros((len(y_pred_mean),numParams))
  stddev_predictions = np.zeros((len(y_pred_stddev),numParams))
  for i in range(numParams):
    predictions[:,i] = y_pred_mean[:,i]*(minmax[i][1]-minmax[i][0])+minmax[i][0]
    stddev_predictions[:,i] = y_pred_stddev[:,i]*(minmax[i][1]-minmax[i][0])+minmax[i][0]

  paramArrayPredict = omegaParamArrayTrain[0:numPredict][:,:numParams]

  predMinusTruth = predictions - paramArrayPredict

  horizontal_line_value = 0
  #errors = np.squeeze(stddev_predictions)#*np.squeeze(predictions)
  errors = np.squeeze(stddev_predictions*np.mean(np.abs(predMinusTruth)/predictions))
  data_points = np.squeeze(predMinusTruth[0:numDataPoints])

  plt.hlines(horizontal_line_value, 0, len(data_points), color='red')

  plt.errorbar(np.arange(len(data_points)), data_points, yerr=errors[0:numDataPoints], fmt='o')
  plt.title("Predictions with Uncertainty")
  plt.xlabel("Data Point")
  plt.ylabel("Prediction - Truth")
  plt.show()
  if colab == False:
    plt.savefig('./images/pred-truth_'+str(keyword)+'.png')
  else:
    plt.show()


  withinErrorBars = np.abs(np.squeeze(predMinusTruth)) - np.abs(errors)
  mask = withinErrorBars <= 0
  num_elements_leq_zero = np.sum(mask)
  print("The number of data points within error bars: " + str(num_elements_leq_zero))
  print("Percentage within error bars: " + str(num_elements_leq_zero/numPredict*100))

#print(len(trainDataset))

