import numpy as np
from PIL import Image, ImageOps
from os import listdir
import random

hotdogs_path = 'images/hd'
pizzas_path = 'images/pizza'
test_path = 'images/test'

# Get a specific image.
def get_file_data(file_path):
    image = Image.open(file_path).convert('LA').resize((200, 200), Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    data = np.asarray(image)
    data = np.reshape(data, 40000)
    data = data/255.0
    return data

# Get all images from path.
def get_images_from_path(folder_path):
    images = []
    for filename in listdir(folder_path):
        try:
            images.append(get_file_data(folder_path + '/' + filename))
        except IOError:
            print('Found one non image file.')
    return images

# Get train images with labels.
def get_all_image_data():
    data = []
    
    # Get hotdogs.
    for item in get_images_from_path(hotdogs_path):
        data.append([[1, 0], item])

    # Get pizzas.
    for item in get_images_from_path(pizzas_path):
        data.append([[0, 1], item])
    
    # Shuffle images.
    random.shuffle(data)

    return data

# Get test images.
def get_test_images():
    return get_images_from_path(test_path)