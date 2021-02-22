import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_image(image):
    
    """ Plot a given image, (BGR image format expected) """
    
    plt.imshow(convert_to_rgb(image))
    plt.show()

    
def convert_to_rgb(image):
    
    """ Convert the BGR to RGB (BGR image format expected) """
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def show_image(img, title="title"):
    
    """show image in the new window (it holds the thread)"""
    
    cv2.imshow(title,img)
    cv2.waitKey(0)
    
def get_maked_location(image, coord_list):
    
    """ Mask section of the image by coordinate and 
        return 
            crooped : mask with real image section
            bool_mask : mask (2-D numpy array) with True at location and flase otherwise
    """
    mask = np.ones((image.shape),dtype=np.uint8)
    mask.fill(255)
    masked_image = cv2.fillPoly(mask, np.array([coord_list], dtype=np.int32),0) # mark the CR as black (0), every thing else white (255)
    bool_mask = (masked_image == 0)[:,:,0] # we are only intersted in the mask, hence 2D is enough
    cropped = cv2.bitwise_or(image, masked_image) # extract CR rest is white
    
    return cropped, bool_mask


def overlap_bbox(bbox_1, bbox_2):
    
    top_left1, bottom_right1 = bbox_1
    top_left2, bottom_right2 = bbox_2
    
    lower_top_left_y = max(top_left1[1], top_left2[1])
    righter_top_left_x = max(top_left1[0], top_left2[0])
    
    higher_bottom_right_y = min(bottom_right1[1], bottom_right2[1])
    lefter_bottom_right_x = min(bottom_right1[0], bottom_right2[0])
        
    if (lower_top_left_y > higher_bottom_right_y) or (righter_top_left_x > lefter_bottom_right_x):
        return 0.0
    
    overlap = (higher_bottom_right_y - lower_top_left_y)*(lefter_bottom_right_x - righter_top_left_x)
    
    return overlap

def get_pixel_area(top_left_cord, bott_right_cord):
    return (bott_right_cord[0] - top_left_cord[0]) * (bott_right_cord[1] - top_left_cord[1])


def write_list(path, my_list):
    with open(path, 'wb') as fp:
        pickle.dump(my_list, fp)
        
def read_list(path):
    with open (path, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist