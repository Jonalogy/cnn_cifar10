import os
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import numpy as np
import pickle
import random

tflite_model_file = "./models/lite_model.tflite"
cifar_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_test_images ():
  with open('./dataset/cifar10_test_batch', mode='rb') as file:
    # note the encoding type is 'latin1'
    test_ds = pickle.load(file, encoding='latin1')
    return [
      test_ds,
      test_ds['data'].reshape((len(test_ds['data']), 3, 32, 32)).transpose(0, 2, 3, 1),
      test_ds['labels']
    ]

def random_select_image ():
  randomIdx = random.randint(0, len(features)-1)
  testImg = features[randomIdx].astype('float32') ; print(testImg.shape, testImg.max(), testImg.dtype)
  return (testImg, randomIdx)

test_ds, features, labels = load_test_images()
(testImg, randomIdx) = random_select_image()


# im = Image.fromarray(features[randomIdx])
# print(cifar_labels[labels[randomIdx]])

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(testImg, axis=0))

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = cifar_labels[np.argmax(output_data)]

print("Test image filename:", test_ds['filenames'][randomIdx])
print("Test image class:", cifar_labels[test_ds['labels'][randomIdx]])
print("Prediction:", predicted_class)