import pandas as pd
import cv2
import random
import numpy as np
import time
import h5py
import pickle
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

weather_dict = {"c":"Cloudy",
                "r":"Rainy",
                "s":"Sunny"}



#------------------------------------------ GENERIC METHODS--------------------------------------------------------



def convert_to_rgb(image):
    
    """ 
    Convert the BGR to RGB (BGR image format expected) 
    image: Image that needs to be converted to RGB format
    """
    
    return np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def plot_image(image):
    
    """ Plot a given image, (BGR image format expected) """
    
    plt.imshow(np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    plt.show()
    
    
   
    
#------------------------------------- EXPERIMENT SPECEFIC METHODS---------------------------------------------------

def convert_path(path, pds_root_dir):
    
    """This method is used to convert the path saved in pklot dataset csv to path which can be used universally
       path         : Path to be converted
       pds_root_dir : Root folder wher PKLot image dataset is located 
    """
    path = path.replace("\\",'/')
    return pds_root_dir+path.split("Dataset")[1]

def convert_cnr_path(file_path, dir_path):
    
    """
    This method converts the partial path of the image to complete path 
    file_path : Image file path to be converted
    dir_path  : Path till the inside of the CNR image dataset directory
    
    """
    return os.path.join(dir_path, "PATCHES", file_path)


def get_validation_conf_matrix_from_fold(fold_piece, backbone_detector, occlusion_fract, dataset_img_path, banner = ""):
  
    """
    This method execute the detection (classification) for given list of image path using given detection model
    
    fold_piece :        List that contains image path and target labels (sometimes occluson coordinates)
    backbone_detector : Object of Backbone detector class, it contains the detection model
    dataset_img_path :  Location of source dataset, it is used to create relative image path
    banner :            Label to specify the status of the experiment. It is used to monitor the completion status of the long experiment session
    
    """
    
    
    y_list = []                    # Detection output list
    backbone_time_list = []        # Execution time of the backbone network per detection
    classifier_time_list = []      # Time taken by the classifier per detection
    exect_time_list = []           # Total exectution time of detection per detection
    prediction_conf_list = []      # Probability of car's presence in the input image

# Process input data (segregate input and output)

    temp = np.array(fold_piece)
    x = temp[:,0].tolist()         # Input image path
    t = temp[:,1]                  # Target/ output label
    
# Get occlusion box coordinates if required
    if occlusion_fract != None:
        occ_col_ind = get_index_from_fract(occlusion_fract) + 4 # First four values doesn't belong to coordinates 
#         print(temp.shape)
#         print(occ_col_ind)
        occ_cordinates = temp[:,occ_col_ind].tolist()
        
# Iterate over the list of images
    for ind, img_path in enumerate(x):

# Convert to relative path and read input image
        if "cnr" in dataset_img_path.lower():
            img_temp = cv2.imread(convert_cnr_path(img_path, dataset_img_path))
        else:
            img_temp = cv2.imread(convert_path(img_path, dataset_img_path))
            
            
# Add occlusion box on the image if needed       
        if occlusion_fract != None:
            occ_cord = occ_cordinates[ind]
            img_temp = get_occluded_img_from_coordinates(occ_cord, img_temp)
            
        sp = time.time()                         # starting point of the detection process
        img_temp = convert_to_rgb(img_temp)      # convert image from bgr (opencv) format to rgb
        
# Main detection (CHANGE THIS CODE different method being called)
        pred_list, backbone_time, prediction_conf = backbone_detector.prediction_method([img_temp])
    
#Save detection output     
        exect_time_list.append(time.time() - sp)
        pred = pred_list[0]
        backbone_time_list.append(backbone_time)
        prediction_conf_list.append(prediction_conf)
        y_list.append(pred)
        
# Execution monitoring
        done = ((ind+1)/len(x))*100
        second_left = np.mean(exect_time_list)*(len(x)-ind-1)
        print(banner)
        print("Percentage done :"+str(done))
        print("Minuites left :"+str(second_left/60))
        clear_output(wait=True)
        
# Final output of the detection
    conf_matrix = confusion_matrix(t, y_list, labels=['Empty', 'Occupied']) # Confusion matrix array of the detection output
    report = classification_report(t, y_list, labels=['Empty', 'Occupied']) # Classification report of the detection output
    accuracy = accuracy_score(t, y_list)
    
    return conf_matrix, report, accuracy, y_list, prediction_conf_list, np.mean(backbone_time_list), np.mean(exect_time_list)




def write_response_in_file(file_name, confusion_matrix, report, accuracy, y_list, confidence_list, bacbone_time, total_detect_time, name):
    
    """
    This method is used to save the progress of the experiment during the execution. It saved the progress in 3 different files
    
    """
    
    list_file = file_name+"_list.dict" # Each entry contained list of detection output [prediction , confidence] of all image detection in the experiment
    dict_file = file_name+".dict"      # Each enrty contained list of detection analysis [accuracy confusing matrix etc]
    text_file = file_name+".txt"       # This file is updated for monitoring purpose during experiment execution
    
# Updating monitoring file
    
    write_in_file(text_file, name)
    write_in_file(text_file, "Accuracy : "+str(accuracy))
    write_in_file(text_file, "Backbone Time : "+str(bacbone_time))
    write_in_file(text_file, "Total Detection Time : "+str(total_detect_time))
    write_in_file(text_file, str(confusion_matrix))
    write_in_file(text_file, "Mean_conf : "+str(np.mean(confidence_list)))
    write_in_file(text_file, str(report))
    write_in_file(text_file, "---------------\n\n\n")
    
# Saving expreimental result    
    temp = [accuracy, bacbone_time, total_detect_time, confusion_matrix, np.mean(confidence_list)]
    second_dict = [y_list, confidence_list]
    
    update_dict(dict_file, name, temp)
    update_dict(list_file, name, second_dict)
    
    

        
 
#----------------------------  METHODS RELATED TO SAVING FILES AND CREATING DIRECTORIES---------------------------------


def create_directory(model_name, path):
    
    """
    This method creates directory for the model at a given path (Used for preprocessing)
    model_name : Name of the model
    path       : Path where directory needs to be build
    """
    model_path = os.path.join(path, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path

        
def update_dict(path, key, update):
    
    """
    This method update the file containing experiment results during the experiments by updating the dictionary saved in the file
    
    path   : Location of the file
    key    : Key value of the dictionary
    Update : The value to be mapped to the key
    """
    if (not os.path.isfile(path)):
        write_list(path, {key:update})
    else:
        temp_dict = read_list(path)
        temp_dict[key] = update
        write_list(path, temp_dict)
        
        

def write_array(path, list_arr):
    
    """
    This method writes list of array to the file
    path     : Location of the file
    list_arr : Array that needs to be stores
    """
    
    with h5py.File(path, 'w') as hf_object:
        hf_object.create_dataset(name='array', data=list_arr, compression="gzip")
        hf_object.close()
        
def read_array(path):
    
    """
    This methods reads the array at given path and returns it
    path : Location of the array
    """
    
    hf_object = h5py.File(path, 'r')
    ret_obj = np.array(hf_object["array"])
    hf_object.close()
    return ret_obj

def write_in_file(file_path, text):
    
    """
    This method is used to update the status of execution of the experiment for monitoring by writing the status in the text file
    file_path : Path of the text file
    text : string of text
    """
    
    result_file = open(file_path,"a+")
    result_file.write(text+"\n")
    result_file.close()


        
def save_k_fold_dataset(path, data):
    """
    It saves dictionary of image paths segmented in folds into a pickle file (Preprocessing)
    path : Location of the pickle file
    data : Data to be saved in the pickle file
    """
    with open(path, 'wb') as filename:
        pickle.dump(data, filename)
        
def read_k_fold_dataset(path):
    
    """
    It reads the pickle file at given location (Preprocessing)
    """
    
    with open(path, 'rb') as filename:
        temp_fold_data_new = pickle.load(filename)
        return temp_fold_data_new
    
    
    
def write_list(path, my_list):
    
    """
    Write a lict into pickle file
    Path    : path to the pickle file
    my_list : list to be saved
    """
    with open(path, 'wb') as fp:
        pickle.dump(my_list, fp)
            
def read_list(path):
    
    """
    Read the list from the given pickle file
    path : path to the pickle file
    """
    with open (path, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist






#------------------------------ METHODS RELATED TO OCCLUSION BOX CREATION-------------------------------------------


def get_index_from_fract(fract):
    """" 
    This method maps fraction of length for occlusion to index in the list of occlusion (columnlist)
    fract : fraction that needs to be converted to index according to the occlusion predefined occlusion pararmeters 
    """
    """These values were picked while creating dataset for occlusion experiments"""
    init_occlusion_fract = 0.43 
    occlusion_increment = 0.01
    
    return round((float(fract) - init_occlusion_fract)/occlusion_increment)



def make_noise_occlusion_box(height, width):
    """
    It creates an image array of noise (occlusion box) of given dimensions and return it
    height : Height of the occlusion box
    width  : Width of the occlusion box
    """
    ret_img = np.zeros((height, width, 3))
    for c in range(3):
        d2_list = []
        for i in range(height):
            temp_noise = np.zeros((height, width,3))
            temp_list = [255]*(width//2)
            temp_list.extend([0]*(width - (width//2)))
            random.shuffle(temp_list)
            d2_list.append(temp_list)
        ret_img[:,:,c] = np.array(d2_list)
        
    return  ret_img




def get_occ_box_from_coordinates(tl, br):
    
    """
    It creates a box of noise/occlusion by using top left and bottom right pixel coordinates
    tl : Top Left pixel coordiates as tuple (x,y)
    br : Bottom Right pixel coordiates as tuple (x,y)
    """
    
    width = br[0]-tl[0]
    height = br[1]-tl[1]
    return make_noise_occlusion_box(height, width)

def get_occluded_img_from_coordinates(coordinate_tuple, img):
    
    """
    It uses pixel coordinates and unoccluded image and returns an image with occlusion box at the pixel location
    coordinate_tuple = It contains two tuples top left and bottom right pixel coordinates of the occlusion box (TL_(x,y), BR_(x,y))
    
    img = unoccluded original image
    
    """
    
    tl = coordinate_tuple[0]
    br = coordinate_tuple[1]
    occlusion_box = get_occ_box_from_coordinates(tl, br)
    img[tl[1]:br[1],tl[0]:br[0],:] = occlusion_box
    
    return img



def get_occlusion_coordinates(box_height, box_width, img_height, img_width):
    """
    It is used to find the an appropiate location for the occlusion box  accoriding to given dimensions of the image and occlusion box
    
    box_height :  Height of the occlusion box image
    box_width  :  Width of the occlusion box image
    img_height :  Height of the original image
    img_width  :  Width of the original image
    """
    
    bottom_right_y = random.randint(box_height,img_height)
    bottom_right_x = random.randint(box_width,img_width)
    top_left_x = bottom_right_x - box_width
    top_left_y = bottom_right_y - box_height
    
    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)




    
#------------------- METHODS USED WHILE EARLY PREPROCESSING--------------------------------------------------
    

def get_k_fold_dataset(pk_lot_dataset ,k):
    
    "create k fold dataset (path of images) for given dataset"
    
    seed_value = 9
    random.seed(seed_value)
    
    weather = ["Cloudy", "Rainy", "Sunny"]
    k_fold_dict = {}
    for each_weather in  weather:
        selected_ds = pk_lot_dataset[["Image_path","Label"]][(pk_lot_dataset["Weather"] == each_weather)]
        suffled_list = list(zip(selected_ds["Image_path"].tolist(), selected_ds["Label"].tolist()))
        random.shuffle(suffled_list)
        seg_size = int(len(suffled_list)/k)
        fold_list = []
        for i in range(k-1):
            fold_list.append(suffled_list[(i*seg_size):((i*seg_size)+seg_size)]) 
        fold_list.append(suffled_list[((k-1)*seg_size):])
        k_fold_dict[each_weather] = fold_list
        
    return k_fold_dict



def small_input(train_x, train_t, val_x, val_t, size = 2500):
    
    """
    This method reduced the size of the training and validation data (Used for preprocessing)
    size : reduced size (no of rows/images)
    """
    
    temp = train_x[:size]
    temp.extend(train_x[:size])
    train_x = temp

    temp = list(train_t[:size])
    temp.extend(list(train_t[-size:]))
    train_t_arr = np.array(temp).reshape((len(temp), 1))
    
    for weather in weather_dict.values():
        val_x[weather] = val_x[weather][:size]
        val_t[weather] = val_t[weather][:size]
    
    return train_x, train_t_arr, val_x, val_t



def print_shapes(train_x, train_t_arr, validation_x, validation_t, feature_vect):
    
    """ Mehtod was used for monitoring purpose (Preprocessing)"""

    print("Training X : ",len(train_x))
    print("After Feature_Extraction :",(len(train_x), feature_vect))
    print("Training T : ",train_t_arr.shape)

    for weather in weather_dict.values():
        val_x  = validation_x[weather]
        val_t_arr  = validation_t[weather]
        print("----"+weather+"----")
        print("Validation X : ",len(val_x))
        print("Validation T : ",val_t_arr.shape)
        print("After Feature_Extraction :",(len(val_x), feature_vect))
        
        
        
        
def create_k_fold_set(k_fold, pk_lot_csv_path):
    
    
    """
    It reads the image path in the given csv file and divide the entire dataset into smaller chunks/folds categorized weatherwise. (Preprocessing)
    
    k_fold          : No of folds/ pieces required
    pk_lot_csv_path : Path to the csv file
    
    """
    
    k_folded_dataset = get_k_fold_dataset(pk_lot_dataset = pd.read_csv(pk_lot_csv_path), k=k_fold)
    temp_fold_data = []

    for kind in range(k_fold):
        training = []
        validation_dict = {}
        for wth in k_folded_dataset.keys():
            k_fold_copy = k_folded_dataset[wth].copy()
            validation_dict[wth] = k_fold_copy.pop(kind)
            for kw in k_fold_copy:
                training.extend(kw)

        train_x = []
        train_t = []

        for train in training:
            train_x.append(train[0])
            train_t.append(train[1])

        train_t_arr = np.array(train_t).reshape((len(train_t), 1))

        validation_x = {}
        validation_t = {}

        for k in list(validation_dict.keys()):
            vaidation = validation_dict[k]
            val_x = []
            val_t = []
            for valid in vaidation:
                val_x.append(valid[0])
                val_t.append(valid[1])
            validation_x[k] = val_x
            validation_t[k] = np.array(val_t).reshape((len(val_t), 1))

        temp_fold_data.append([train_x, train_t_arr, validation_x, validation_t])

    training = None
    vaidation = None
    
    return temp_fold_data




# def print_validation_result(result_file_path, conf_matrix, total_mean_time, mean_of_all_times, k):
    
#     """Used for monitoring purpose"""

#     result_file = open(result_file_path,"a+")
#     result_file.write("-----------\n")
#     result_file.write("Confusion Matrix\n")
#     result_file.write(str(conf_matrix))
#     result_file.write("\n")
#     result_file.write("Execution Time : "+k+"\n")
#     result_file.write("Mean of forward pass                  : "+str(total_mean_time)+"\n")
#     result_file.write("Mean of Feature, Backbone, Classifier : "+str(mean_of_all_times)+"\n")
#     result_file.close()


    
    
# def add_noise_to_img(img, occlusion_fraction, bgr_code = (255,255,255)):
#     img_height, img_width = img.shape[:2]
#     box_height = round(img.shape[0] * occlusion_fraction)
#     box_width = round(img.shape[1] * occlusion_fraction)
#     bottom_right_y = random.randint(box_height,img_height)
#     bottom_right_x = random.randint(box_width,img_width)
#     top_left_x = bottom_right_x - box_width
#     top_left_y = bottom_right_y - box_height
#     img = cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color = bgr_code, thickness = -1)
    
#     return img







        