# EdgeTpu
# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from periphery import GPIO
import numpy

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input',
                      help='Image to be classified.')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-k', '--top_k', type=int, default=1,
                      help='Max number of classification results')
  parser.add_argument('-t', '--threshold', type=float, default=0.0,
                      help='Classification score threshold')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {} #reading the label file generated

  interpreter = make_interpreter(*args.model.split('@')) # Models obtained from TfLiteConverter is run in Python with Interpreter.

  interpreter.allocate_tensors()  
 ## tflite_model can be saved to a file and loaded later, or directly into the Interpreter. 
##Since TensorFlow Lite pre-plans tensor allocations to optimize inference, the user needs to call allocate_tensors() before any inference.

  _, height, width = interpreter.get_input_details()[0]['shape']
  size = [height, width] # getting the size of the input 

  trigger = GPIO("/dev/gpiochip2", 13, "out")  # pin 37 defining trigger

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  #for i in range(1,351):
  while 1:
    #input_image_name = "./testSample/img_"+ str(i) + ".jpg"
    #input_image_name = "./testSample/img_1.jpg"
    #image = Image.open(input_image_name).resize(size, Image.ANTIALIAS)
    arr = numpy.random.randint(0,255,(28,28), dtype='uint8')  ##creating 28 numpy arrays with 28 elements
    image = Image.fromarray(arr, 'L').resize(size, Image.ANTIALIAS) # picking up an array out of 28 and resizing based on the input size
    common.set_input(interpreter, image) # setting the selected resized image as the input

    start = time.perf_counter() # starting the time for inference
    trigger.write(True) # setting trigger
    interpreter.invoke() # running the inference
    trigger.write(False) # setting trigger low after the inference is done
    inference_time = time.perf_counter() - start # eveluating the inference time
    print('%.6fms' % (inference_time * 1000))  
    
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)

    print('RESULTS for image ', 1)
    for c in classes:
      print('%s: %.6f' % (labels.get(c.id, c.id), c.score))
    #time.sleep(2)


if __name__ == '__main__':
  main()
