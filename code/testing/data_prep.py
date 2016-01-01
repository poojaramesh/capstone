from os import listdir, makedirs, walk
from os.path import join, isdir 
import numpy as np
from skimage import io
from itertools import izip
import math
from random import shuffle

def sampling_data(data_path, sample_size=-1):
    '''
        Randomly samples images from a given folder 
        Input:  path - path to image folder (string)
                sample_size - no.of images to be sampled (int)
        Output: Array of sampled image names
    '''
    path, dirs, files = walk(data_path).next()
    if sample_size < 0:
        sample_size = len(files)
    frame_sample = np.random.choice(files, sample_size, False)
    frame_path = [join(path, f) for f in frame_sample]
    return frame_path 

def get_label(frame_name):
    '''
        Assigns a target label to each image. 0 for non-mitosis, 1 for mitosis, 2 for background 
        Input: frame_name - image filename (string)
        Output: label (int)
    '''
    if "background" in frame_name:
        label = 2
    if "non" in frame_name:
        #label = "non_mitosis"
        label = 0
    else:
        #label = "mitosis"
        label = 1
    return label

def get_counts(image_list, nb_classes):
    '''
        Finds the distribution of the training set
        Input: 
            image_list - list containing list of images after sampling
            nb_classes - 2 for binary class, 3 for multiclass
    '''
    labels = np.array([get_label(name) for name in image_list]) 
    count_0 = (labels == 0).sum() 
    count_1 = (labels == 1).sum()
    if nb_classes == 3:
        count_2 = (labels == 2).sum() 
        return count_0, count_1, count_2
    else:
        return count_0, count_1

def create_image_list(path, sample_size=[-1, -1, -1, -1, -1], use_mr=True):
    '''
        Creates a list of images (with abolsute paths) depending on sample size for each class
        Input: 
            path - root folder containing images split for each category (string)
            sample_size - number of samples to randomly pick from each class [mitosis, non_mitosis, background, mitosis_mirrored_rotated, non_mitosis_mirrored_rotated]
            use_mr - Set to upsample mitosis and non-mitosis classes (boolean)
        Output: list of images
    '''
    image_list = []
    categories = ['mitosis', 'non_mitosis', 'background']
    if use_mr:
        categories = categories + ['mitosis_mr', 'non_mitosis_mr']
    
    for i, category in enumerate(categories):
        image_list = image_list + sampling_data(join(path, category), sample_size[i]) 

    shuffle(image_list)
    return image_list

def get_input(image_list):
    '''
        Generates the X and y for training by reading each image, scaling and create arrays of shape(x,100,100,3)
        Input: 
            image_list: list of images to use for the training/testing
        Output: X (images as np arrays), y (labels)
    '''
    X, y = [], []
    for image in image_list:
        img = io.imread(image)
        X.append(img)
        y.append(get_label(image))
    X_train = np.array(X).astype('float32')
    X_train /= 255
    Y_train = np.array(y)
    return X_train, Y_train 

def flow(image_list, batch_size=32):
    '''
    Generator that provides sets of data based on the batch_size to the CNN function train_on_batch
    Input:
        image_list - list of images to use for training and testing
        batch_size - data size to feed into the CNN
    Output:
        X, y
    '''
    num_images = len(image_list)
    nb_batch = int(math.ceil(float(num_images)/batch_size))
    for b in range(nb_batch):
        batch_end = (b+1)*batch_size
        if batch_end > num_images:
            nb_samples = num_images - b*batch_size
        else:
            nb_samples = batch_size

        X, y = get_input(image_list[b*batch_size:b*batch_size+nb_samples])
        yield X, y


if __name__ == '__main__':
    path = '/data/ScannerA'
    #for category in ('mitosis', 'non_mitosis'):
    #    mirror_rotate_images(path, category, rotate=True, mirror=True)

    X, y = get_input('/data/ScannerA', [1000, 1000, 4000], True)
