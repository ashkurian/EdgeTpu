# EdgeTpu

# EdgeTpu
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)


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
input_size = 384
node_size = 10
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
  tf.keras.layers.InputLayer(input_shape=(input_size)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(node_size)
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


###setting weights
w = np.zeros((input_size,node_size),dtype='int8')
for i in range(input_size):
  for j in range(node_size):
      w[i][j] = 97
  
b = np.zeros((node_size,),dtype='int8')
for i in range(node_size):
  b[i] = 26
wb = [w,b]
wb
model.set_weights(wb)
model.get_weights()


###integer quantization 

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
#### generating the tflite file
tflite_model_quant = converter.convert()


################# downloading the generated .tflite file 
with open('1L10Nallweights97.tflite', 'wb') as f:
  f.write(tflite_model_quant)
  # Save the quantized model to file to the Downloads directory
# Download the digit classification model
from google.colab import files
files.download('1L10Nallweights97.tflite')

print('`1L10Nallweights97.tflite` has been downloaded')
