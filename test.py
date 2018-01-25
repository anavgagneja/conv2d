from PIL import Image
from conv import Conv2D
import numpy as numpy
import torch 

imageOne = './images/image01.jpg'
imageTwo = './images/image02.jpg'

def image_to_array(image_path):
    img = Image.open(image_path)
    img.load()
    img_array = numpy.asarray(img)
    print(img_array.shape)
    return img_array

img_array1 = image_to_array(imageOne)
img_tensor1 = torch.from_numpy(img_array1)
img_array2 = image_to_array(imageTwo)
img_tensor2 = torch.from_numpy(img_array2)

task1 = Conv2D(3, 1, 3, 1, 'known')
task3 = Conv2D(3, 3, 3, 2, 'known')
task2 = Conv2D(3, 2, 5, 1, 'known')

task1.forward(img_tensor1)
task3.forward(img_tensor1)
task2.forward(img_tensor1)

task1.forward(img_tensor2)
task3.forward(img_tensor2)
task2.forward(img_tensor2)