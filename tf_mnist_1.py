# Kelvin Filyk
# May 14th 2019
# Originally taken from https://www.tensorflow.org/tutorials. Used to learn how Keras implements machine learning functionality.

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist # taken from keras-builtin-datasets

(x_train, y_train),(x_test, y_test) = mnist.load_data() # save training, testing data to different tuples
print("\nExample pixel value: ")
print(x_train[0]) # Each pixel has a range between 0 and 255 initially.
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize all values between 0 and 1.
print("Type: ")
print(type(x_train)) # A numpy.ndarray (n-dimentional array).
#print("Functions of Module: ")
#print(dir(x_train))
print("Shape: ")
print(x_train.shape) # 60000x28x28
print("Data type: ")
print(x_train.dtype) # float64

# https://keras.io/models/sequential/
# Sequentail model: A linear stack of layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Collapses multiple dimensions into a single dimension (from a 2D 28x28 array to a 1D column of size 784).
  tf.keras.layers.Dense(512, activation=tf.nn.relu, use_bias=True),
  # Creates a Dense layer of neurons with 512 nodes of type rectified linear unit.
  # Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation
  # function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created
  # by the layer (only applicable if use_bias is True). Outputs arrays of shape (*, 512)
  tf.keras.layers.Dropout(0.2),
  # https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/.
  # Creates a Dropout layer where neurons have a 20% chance of being dropped.
  # The outputs/activations of the first Dense layer (of size 512) have dropout applied to them,
  # thus affecting the weights of the second Dense layer (of size 10).
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  # https://en.wikipedia.org/wiki/Softmax_function
])

model.compile(optimizer='adam', #??
              loss='sparse_categorical_crossentropy', #??
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
