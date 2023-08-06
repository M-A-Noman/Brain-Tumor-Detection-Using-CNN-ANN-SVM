# imports for CNN
import os
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt 
from tkinter import filedialog
from PIL import Image,ImageTk
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow._api.v2.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
cnn = Sequential() 
from keras_preprocessing import image
import matplotlib.image as mpimg

# imports for ANN
import cv2
import numpy as np
import os
model = Sequential()
data_dir = 'xray/train/'
categories = ["NORMAL", "Tumor"]
img_size = 128
from keras.models import Sequential
from keras.layers import Dense
