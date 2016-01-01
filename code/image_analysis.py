'''
    Adds each sub_image into Mitosis/Background/Non-Mitosis folders 
'''

import numpy as np
import pandas as pd
from os.path import join, isdir, exists
from os import listdir, stat, makedirs, walk
from skimage import io

def get_mitosis_locations(path, class_type):
    '''
        Creates a dataframe that has the pixel locations for the (non)mitosis in each image for a given scanner type
        Input:  path to scanner folder (string)
                class_type (int)
                    2 for mitosis (# of _ in the filename)
                    3 for non_mitosis (# of _ in the filename)
        Output: dataframe consisting of (non)mitosis locations
    '''
    df_mitosis = pd.DataFrame(columns=['row', 'col', 'd', 'name'])
    scanner_dir = [f for f in listdir(path) if isdir(join(path, f))]
    for d in scanner_dir:
        dir_path = join(path, d, "mitosis")
        if isdir(dir_path):
            for f in listdir(dir_path):
                if f.count('_') == class_type and "_mitosis.csv" in f:
                    file_path = join(dir_path, f)
                    if stat(file_path).st_size > 0:
                        df_temp = pd.read_csv(file_path, names=['col', 'row', 'd'])
                        file_name = f.split('_')
                        df_temp['name'] = file_name[0] + '_' + file_name[1]
                        df_mitosis = pd.concat([df_mitosis, df_temp], axis=0, ignore_index=True) 
    return df_mitosis

def get_valid_frames(df_mitosis, df_non_mitosis):
    '''
        Returns a list of all frames that have either have a mitosis or non_mitosis occurence in them
        Input: mitosis/non_mitosis (dataframe)
        Output: names of all frames (set)
    '''
    mitosis_frames = set(df_mitosis.name.unique())
    non_mitosis_frames = set(df_non_mitosis.name.unique())
    all_frames = mitosis_frames.union(non_mitosis_frames)
    return all_frames 


def create_imageset(path, xstep, ystep, win_size, cell_distance):
    '''
        Imports each image from a given scanner type and segments each frame image
        Input:  all_frames - list of all frames that either have a mitosis/non_mitosis occurence in them (set)
                path - path to the frames that need to be segmented (string)
                xstep - scanning window step size (int)
                ystep - scanning window step size (int)
                win_size - scanning square window size (int)
        Output: none
    '''
    print "Creating dataframe of mitosis locations"
    df_mitosis = get_mitosis_locations(path, 2) 
    print "Creating dataframe of non_mitosis locations"
    df_non_mitosis = get_mitosis_locations(path, 3)
    all_frames = get_valid_frames(df_mitosis, df_non_mitosis)
    
    scanner_dir = [f for f in listdir(path) if isdir(join(path, f))]
    for d in scanner_dir:
        dir_path = join(path, d, "frames", "x40")
        print "Splitting images in {}".format(dir_path)
        if isdir(dir_path): 
            for frame in listdir(dir_path):
                print "Splitting frame {}".format(frame)
                frame_name = frame.split('.')[0]
                if frame_name in all_frames:
                    frame_path = join(dir_path, frame)
                    segment_images(frame_name, frame_path, path, xstep, ystep, win_size, df_mitosis, df_non_mitosis, cell_distance)


def segment_images(frame_name, frame_path, split_path, xstep, ystep, win_size, df_mitosis, df_non_mitosis, cell_distance):
    '''
        Segments a given frame into sub-images based on window and step size
        Input:  frame_name - name of the frame to be segmented (string)
                frame_path - path to the frame (string)
                split_path - location where segmented images are saved (string)
                xstep - scanning window step size (int)
                ystep - scanning window step size (int)
                win_size - scanning square window size (int)
                df_mitosis - mitosis occurence co-ordinates (dataframe)
                df_non_mitosis - non_mitosis occurence co-ordinates (dataframe)
        Output: None
    '''
    img = io.imread(frame_path)
    img_x, img_y = img.shape[0], img.shape[1]
    
    img = extend_image(img, win_size, xstep, ystep)
    
    for x_cord in range(0, img_x, xstep):
        for y_cord in range(0, img_y, ystep):
            label = get_image_label(frame_name, x_cord, y_cord, x_cord+win_size, y_cord+win_size, df_mitosis, df_non_mitosis, cell_distance)
            frame = frame_name + "_" + str(x_cord) + "_" + str(y_cord) + "_" + label + ".tiff"
            save_path = join(split_path, label, frame)
            xlim = x_cord + win_size
            ylim = y_cord + win_size
            io.imsave(fname=save_path, arr=img[x_cord:xlim, y_cord:ylim, :])


def extend_image(img, win_size, x_step, y_step):
    '''
        Extends the image based on the window size to have evenly sized sub-images by mirroring the image across the bottom and right edges
        Input:  img - array of rgb values for an image (numpy array)
                win_size - scanning square window size
        Output: numpy array
    '''
    img_x, img_y = img.shape[0], img.shape[1]
    mod_x = img_x % (x_step)
    mod_y = img_y % (y_step)
    
    extend_x = win_size - mod_x
    img = np.vstack((img, img[img_x-extend_x-1: img_x-1, :, :][::-1]))       
   
    extend_y = win_size - mod_y
    img = np.hstack((img, img[:, img_y-extend_y-1:img_y-1, :][:,::-1,:]))
    return img


def get_image_label(frame_name, x1, y1, x2, y2, df_mitosis, df_non_mitosis, cell_distance):
    '''
        Assigns a label to each image if a mitosis or non_mitosis occurence is found within 50px away from the center
        Input:  frame_name (string)
                x1, y1, x2, y2 - cordinates of the left top and right bottom corners of the scanning window (int)
                df_mitosis - mitosis occurence co-ordinates (dataframe)
                df_non_mitosis - non_mitosis occurence co-ordinates (dataframe)
        Output: label mitosis/non_mitosis/background (string)
    '''
    center_x = (x1+x2)/2.
    center_y = (y1+y2)/2.
    df_m =  df_mitosis[df_mitosis.name == frame_name]
    df_nm =  df_non_mitosis[df_non_mitosis.name == frame_name]
    m_count = (abs(df_m.row - center_x) <= cell_distance) & (abs(df_m.col - center_y) <= cell_distance)
    nm_count = (abs(df_nm.row - center_x) <= cell_distance) & (abs(df_nm.col - center_y) <= cell_distance)
    if m_count.sum() > 0:
        return "mitosis"
    elif nm_count.sum() > 0:
        return "non_mitosis"
    else:
        return "background"


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
if __name__ == '__main__':
    # Location to the ScannerA data
    #path = '/Users/pooja/Documents/Capstone/Mitosis/2014/Training/ScannerA'
    #path = '/Users/pooja/Documents/gitRepos/capstone/test_images/ScannerA'
    #create_imageset(path, 50, 50, 100, 40)
    
    path = '/data/ScannerA'
    #path = '/data/Test/ScannerA'
    create_imageset(path, 50, 50, 100, 20)
