import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import pathlib

import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
assert float(tf.__version__[:3]) >= 2.3

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

print(tf.__version__)
input_size = 784
node_size = 500
# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = np.reshape(train_images, (train_images.shape[0], 784))[:,0:input_size]
test_images = np.reshape(test_images, (test_images.shape[0], 784))[:,0:input_size]

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) 
test_images = test_images.astype(np.float32) 
# Define the model architecture
model = tf.keras.Sequential([
  #tf.keras.layers.InputLayer(input_shape=(input_size)),
  tf.keras.layers.InputLayer(input_shape=(28,28)),
  #tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
  tf.keras.layers.Flatten(),
  # tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),

  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),
  # tf.keras.layers.Dense(1),

  #tf.keras.layers.Dense(node_size)
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),

  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),
  tf.keras.layers.Dense(500),

  # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  # tf.keras.layers.Flatten(),
  # tf.keras.layers.Dense(10)
])




# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
#model.fit(
 #train_images,
 #train_labels,
  #  epochs=5,
   #validation_data=(test_images, test_labels)
# )
model.summary()
#w,b = layer1.get_weights()
w = np.zeros((input_size,node_size),dtype='int8')
for i in range(input_size):
  for j in range(node_size):
    w[i][j] = 97
  
    
b = np.zeros((node_size,),dtype='int8')
for i in range(node_size):
  b[i] = 26
wb = [w,b]
#wb
#784 500 configuration for wb

model.layers[1].set_weights(wb)



w1 = np.zeros((node_size,node_size),dtype='int8')
for i in range(node_size):
  for j in range(node_size):
    w1[i][j] = 97
  
wb1 = [w1,b]
for i in range(19):  #19 layers with 500 500 configuration for w1 
  model.layers[i+2].set_weights(wb1)
  #model.layers[i+1].get_weights()



def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
##
converter.target_spec.supported_types = [tf.int8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

#interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
#input_type = interpreter.get_input_details()[0]['dtype']
#print('input: ', input_type)
#output_type = interpreter.get_output_details()[0]['dtype']
#print('output: ', output_type)
#interpreter.get_tensor_details()
#tflite_model_quant_file = 20L500Nmodified.tflite
#20L500Nmodified.write_bytes(tflite_model_quant)


tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"20L500N_mod_mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
#tflite_model_quant
from google.colab import files
files.download(tflite_model_quant_file)

print('`20L500N_mod_mnist_model_quant.tflite` has been downloaded')
