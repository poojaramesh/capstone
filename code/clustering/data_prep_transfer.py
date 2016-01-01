from os import listdir, makedirs, walk
from os.path import join, isdir 
import numpy as np
from skimage import io
from itertools import izip
import math
from random import shuffle
import cPickle as pickle

def mirror_rotate_images(path, category, rotate=True, mirror=True):
    '''
        Creates copies of the images in the folder by rotating and mirroring
        Descends in each folder based on the catergory
        Input:  path - path to root folder (string)
                catergory - mitosis/non_mitosis (string)
                rotate - rotates image counter clockwise 90,180,270 (boolean)
                mirror - mirrors image LR and UP (boolean)
        Output: None 
    '''
    path_category = join(path, category)
    for f in listdir(path_category):
        img_path = join(path_category, f)
        img = io.imread(img_path)
        f_name = f.split('.')[0]
        if rotate:
            rotate_images(f_name, img, path, category)
        if mirror:
            mirror_images(f_name, img, path, category)

def rotate_images(frame, img, path, category):
    '''
        Rotates the image 90,180,270 counter clockwise and saves each version in the folder (non_)mitosis_mr
        Input:  path - path to root folder (string)
                catergory - mitosis/non_mitosis (string)
                frame - name of the frame/image (string)
                img - array of the frame/image (numpy array)
        Output: None 
    '''
    path =  join(path, category+'_mr')
    if not isdir(path):
        makedirs(path)
    for i in range(1,4):
        save_path = join(path, frame+ '_rot' + str(90*i) + '.tiff')
        io.imsave(save_path, np.rot90(img, i))  

def mirror_images(frame, img, path, category):
    '''
        Mirrors the image LR and UP and saves each version in the folder (non)_mitosis_mr
        Input:  path - path to root folder (string)
                catergory - mitosis/non_mitosis (string)
                frame - name of the frame/image (string)
                img - array of the frame/image (numpy array)
        Output: None 
    '''
    path =  join(path, category+'_mr') 
    if not isdir(path):
        makedirs(path)
    save_path = join(path, frame+ '_mirrLR.tiff')
    io.imsave(save_path, np.fliplr(img)) 

    save_path = join(path, frame+ '_mirrUD.tiff')
    io.imsave(save_path, np.flipud(img)) 

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
        Assigns a target label to each image. Multiclass problem 2 for background, 0 for non-mitosis, 1 for mitosis 
        Input: frame_name - image filename (string)
        Output: label (int)
    '''
    if "background" in frame_name:
        #label = "background"
        label = 2
    elif "non" in frame_name:
        #label = "non_mitosis"
        label = 0
    else:
        #label = "mitosis"
        label = 1
    return label

def create_image_list(path, sample_size=[-1,-1,-1, -1, -1], use_mr=True):
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
    labels = np.array([get_label(name) for name in image_list]) 


def get_image_clusters(image_name_path, clusters_path):
    with open (image_name_path) as f:
        image_names = pickle.load(f)
    with open (clusters_path) as f1:
        clusters = pickle.load(f1)
    return image_names, clusters

def create_image_clusters(image_list, image_names, clusters, cluster):
    image_clusters = image_names[clusters == cluster]
    image_list_cluster = [path[0] for name in image_clusters for path in izip(image_list) if name in path[0]]
    return image_list_cluster 

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

