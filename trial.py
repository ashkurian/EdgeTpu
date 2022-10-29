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

  labels = read_label_file(args.labels) if args.labels else {}

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()

  _, height, width = interpreter.get_input_details()[0]['shape']
  size = [height, width]

  trigger = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
  # UART3, 9600 baud
  uart1 = Serial("/dev/ttymxc0", 115200)
  #input_details = interpreter.get_input_details()[0]
  input_details = interpreter.get_input_details()

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  #for i in range(1,351):
  while 1:
    #input_image_name = "./testSample/img_"+ str(i) + ".jpg"
    #input_image_name = "./testSample/img_1.jpg"
    #image = Image.open(input_image_name).resize(size, Image.ANTIALIAS)
    #arr = numpy.random.randint(0,255,(28,28), dtype='uint8')
    arr = uart1.read(784)
    print(list(arr))
    arr = numpy.array(list(arr), dtype='uint8')
    arr = numpy.reshape(arr, (28,28))
    #image = Image.fromarray(arr, 'L').resize(size, Image.ANTIALIAS)
   
    #common.set_input(interpreter, image)
    interpreter.set_tensor(input_details[1], arr)
    #inspector_start = int.from_bytes(uart3.read(1, 1), 'big')
    #print("read {:d} bytes: _{:s}_".format(len(inspector_start), inspector_start))
    #print("Start Signal:", inspector_start)
    start = time.perf_counter()
    trigger.write(True)
    interpreter.invoke()
    trigger.write(False)
    inference_time = time.perf_counter() - start
    output_tensor = interpreter.get_tensor(1)[0]
    uart1.write(output_tensor.tobytes())
    print('%.6fms' % (inference_time * 1000))
    
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)

    #print('RESULTS for image ', 1)
    for c in classes:
      print('%s: %.6f' % (labels.get(c.id, c.id), c.score))
    #time.sleep(2)


if __name__ == '__main__':
  main()
