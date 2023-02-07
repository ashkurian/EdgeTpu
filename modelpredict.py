"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
python3 classify-image.py \
  --model mnist_model_quant_edgetpu.tflite  \
  --labels mnist_labels.txt \
```
"""

import argparse
import time
from periphery import GPIO, Serial
import numpy as np

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model to use')
args = parser.parse_args()

# Load the TFLite model
#interpreter = tf.lite.Interpreter(model_path="/content/mobilenet_v1_1.0_224_quant_edgetpu.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare the input data
input_data = np.random.rand(224, 224, 3).astype(np.uint8)
input_data = np.expand_dims(input_data, axis=0)# add a new dimension for batch size
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the model
interpreter.invoke()

# Get the output data
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
