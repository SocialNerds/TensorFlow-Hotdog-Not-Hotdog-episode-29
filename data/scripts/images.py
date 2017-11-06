import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

hotdogs_path = "images/hd"
cars_path = "images/car"

def get_file_data(file_path):
    data = np.asarray(Image.open(file_path).resize((200, 200), Image.ANTIALIAS))
    data = np.reshape(data, 120000)
    data = data/255.0
    return data

def get_all_image_data():
    # Open all images.
    images = []
    # Get hotdogs.
    for filename in listdir(hotdogs_path):
        try:
            images.append([1, get_file_data(hotdogs_path + "/" + filename)])
        except IOError:
            print("Found one non image file")
    # Get cars.
    for filename in listdir(cars_path):
        try:
            images.append([2, get_file_data(cars_path + "/" + filename)])
        except IOError:
            print("Found one non image file")

    return images

