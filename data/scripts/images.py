import numpy as np
from PIL import Image, ImageOps
from os import listdir
import random

hotdogs_path = "images/hd"
cars_path = "images/car"
test_image = "images/test/test.jpg"

def get_file_data(file_path):
    image = Image.open(file_path).convert('LA').resize((200, 200), Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    data = np.asarray(image)
    data = np.reshape(data, 40000)
    data = data/255.0
    return data

def get_all_image_data():
    # Open all images.
    images = []
    # Get hotdogs.
    for filename in listdir(hotdogs_path):
        try:
            images.append([[1, 0], get_file_data(hotdogs_path + "/" + filename)])
        except IOError:
            print("Found one non image file")
    # Get cars.
    for filename in listdir(cars_path):
        try:
            images.append([[0, 1], get_file_data(cars_path + "/" + filename)])
        except IOError:
            print("Found one non image file")
    
    # Shuffle images.
    random.shuffle(images)

    return images

def get_test_image():
    return get_file_data(test_image)