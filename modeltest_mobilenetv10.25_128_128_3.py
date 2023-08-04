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

  _, height, width,channel = interpreter.get_input_details()[0]['shape']
  #size = [height, width,channel]
  size = [width,height]


  trigger = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
  # UART3, 9600 baud
  uart1 = Serial("/dev/ttymxc0", 115200)
  #input_details = interpreter.get_input_details()[0]
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

   print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')

  #for i in range(1,351):
  while 1:
    #input_image_name = "./testSample/img_"+ str(i) + ".jpg"
    #input_image_name = "./testSample/img_1.jpg"
    #image = Image.open(input_image_name).resize(size, Image.ANTIALIAS)
    #arr = numpy.random.randint(0,255,(28,28), dtype='uint8')
    arr = uart1.read(49152)

    #arr= numpy.random.rand(6336)
    #remove
    #print(list(arr))
    arr = numpy.array(list(arr), dtype='uint8')
    arr = numpy.reshape(arr, (128,128,3))
    arr = numpy.concatenate([arr[numpy.newaxis, :, :]]*1)
    #arr = numpy.expand_dims(arr, axis=0)
    print(arr.shape)
    #interpreter.resize_tensor_input(input_details[0]['index'], (1, 28, 28, 3))
    #common.set_input(interpreter, image)
    interpreter.set_tensor(input_details['index'], arr)
    start = time.perf_counter()

    trigger.write(True)
    interpreter.invoke()
    trigger.write(False)
    inference_time = time.perf_counter() - start
    print(inference_time)
    #output_tensor = interpreter.get_tensor(1)[0]
    output_tensor = interpreter.get_tensor(output_details['index'])[0]
    #print(output_tensor[1])
    #output_tensor[1]=output_tensor[1].tobytes()


    print(output_tensor[1].tobytes())

    #print(list(numpy.array(output_tensor[1])).tobytes())
    uart1.write(output_tensor[1].tobytes())
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)

    #print('RESULTS for image ')
    #for c in classes:
     ## print('%s: %.6f' % (labels.get(c.id, c.id), c.score))
    #time.sleep(2)


if __name__ == '__main__':
  main()
   

    






