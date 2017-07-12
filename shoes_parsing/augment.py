import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import os
import tensorflow as tf
from PIL import Image, ImageStat, ImageEnhance
from random import randint, uniform
from skimage import filters
from skimage.io import imread
from scipy.misc import imsave
import uuid
import sys



model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
graph = tf.get_default_graph()

INPUT_WIDTH = 299
INPUT_HEIGHT = 299

MAX_ROTATION = 40
MIN_BRIGHTNESS = 0.6
MIN_SCALE_PERCENT = 70
MAX_SCALE_PERCENT = 140

out_path = sys.argv[1] + '/'
print(out_path)

preset_labels = ['n04133789', 'n03680355', 'n04200800', 'n04200800', 'n03124043', 'n04120489', 'n04254777', ]

def rescale_to_x(img, x):
    img = img.resize((x, x))
    return img


def listdir_nohidden(path):
    dirs = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            dirs.append(f)
    return dirs


def predict(img):
    img = img.convert('RGB')
    x = np.array(img, dtype = float)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    global graph
    with graph.as_default():
        preds = model.predict(x)
    top3 = decode_predictions(preds,top=20)[0]
    labels = [label for label,description, probability in top3]
    predictions = [(label, description) for label,description, probability in top3]
    for label in labels:
        if label in preset_labels:
            return True
    print('Shoes not found!')
    print(predictions)
    return False


def remove_background(file):
    img = imread(file)
    val = filters.threshold_otsu(img)
    mask = img < val
    img[mask] = 0
    imsave('out.png', img)
    return os.path.abspath('out.png')


def random_crop(img):
    width, height = img.size
    x1 = randint(0, width - INPUT_WIDTH) 
    y1 = randint(0, height - INPUT_HEIGHT)
    return img.crop((x1, y1, x1+INPUT_WIDTH, y1+INPUT_HEIGHT))


def random_scale(img):
    s = randint(MIN_SCALE_PERCENT, MAX_SCALE_PERCENT)/100
    width, height = img.size
    img.resize((int(width*s), int(height*s)))
    return img


def random_rotation(img, bcg):
    angle = randint(-MAX_ROTATION,MAX_ROTATION)
    img = img.convert('RGBA')
    img = img.rotate(angle, resample=Image.BICUBIC)
    out = Image.composite(img, bcg, img)
    return out


def choose_background():
    bg_list = listdir_nohidden('backgrounds')
    i = randint(0,len(bg_list))
    path = 'backgrounds/' + bg_list[i]
    img = Image.open(path)
    return random_crop(img)


def change_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    factor = uniform(MIN_BRIGHTNESS, 1) 
    return enhancer.enhance(factor)


def augment(file, background_swap = False, use_crop = True, use_rotate = True, use_average = True, use_gamma = True, use_scale = True):
    print('Image augmentation')
    if background_swap:
        print('Removing Background')
        path = remove_background(file)
    else:
        path = file
    print('Opening ' + str(path))
    img = Image.open(path)
    mean_color = tuple(map(int, ImageStat.Stat(img).mean))
    print('Image mean: ')
    print(mean_color)
    if use_scale:
        img = random_scale(img)
    if use_rotate:
        print('Rotating')
        if background_swap:
            print('Swapping Background')
            bcg = choose_background()
        elif use_average:
            bcg = Image.new('RGBA', img.size, mean_color)
        else:
            bcg = Image.new('RGBA', img.size, (0,0,0,255))
        img = random_rotation(img, bcg)
    is_shoes = False
    if use_crop:
        print('Cropping')
        i = 0
        while is_shoes is False:
            crop_img = random_crop(img)
            is_shoes = predict(crop_img)
            i += 1
            if i > 30:
                crop_img.show()
                break
    img = crop_img
    if use_gamma:
        print('Gamma')
        img = change_brightness(img)
    return rescale_to_x(img,230)


def process_folder(path):
    save_path = out_path + str(uuid.uuid1()) 
    os.mkdir(save_path)
    print('Processing path: ' + path)
    for file in listdir_nohidden(path):
        file_path = path + '/' + file
        if os.path.isdir(file_path):
            print(file_path + ' is a folder, looking inside of it')
            process_folder(file_path)
        else:
            file_save_path = save_path +  '/' + file
            image = augment(file_path, use_crop = True, use_rotate = False, use_average = False, use_gamma = False, use_scale = False)
            image.save(file_save_path + 'crop_noscale.png')
            image = augment(file_path, use_crop = True, use_rotate = False, use_average = False, use_gamma = False)
            image.save(file_save_path + 'crop.png')
            image = augment(file_path, use_crop = True, use_rotate = True, use_average = False, use_gamma = False)
            image.save(file_save_path + 'crop_rot_noavg.png')
            image = augment(file_path, use_crop = True, use_rotate = True, use_average = True, use_gamma = False)
            image.save(file_save_path + 'crop_rot.png')
            image = augment(file_path, use_crop = True, use_rotate = True, use_average = True, use_gamma = True)
            image.save(file_save_path + 'crop_rot_gamma.png')
            print('Saved: ' + path + 'to:' + file_save_path)



def augment_dataset(start = 0):
    if not os.path.exists(out_path):
        print(out_path + '  does not exist')
        raise
    else:
        print('Already exists: ' + out_path)
    folders = listdir_nohidden('raw_data')
    for i in range(start, len(folders)):
        folder = folders[i]
        path = 'raw_data/' + folder
        print('Processing ' + path)
        process_folder(path)
        with open('start.txt', 'w') as f:
            f.write(str(i) + '\n')
            f.write(folder)


start = 0     
if os.path.isfile('start.txt'):
    with open('start.txt', 'r') as f:
        start = int(f.readline())
augment_dataset(start)







