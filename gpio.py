Example usage:
```
python3 serial-classify.py \
  --model 1L_2500N_allweights97_mnist_model_quant_edgetpu.tflite

  --labels mnist_labels.txt \
```
"""


from periphery import GPIO
import time

led = GPIO("/dev/gpiochip0", 13, "out")  # pin 40
# button = GPIO("/dev/gpiochip4", 13, "in")  # pin 36

try:
  while True:
    led.write(True)
    time.sleep(2)
    led.write(False)
    time.sleep(2)
finally:
  led.write(False)
  led.close()
