import time
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import joblib

class BackboneDetector():
    
    def __init__(self, model_name, root_dir, m2_name = "Mobilenet V2"):
        
        """
        CONSTRUCTOR
            model_name : Name that determine which detection model to be used
            root_dir   : Path to the root directory of the project
            m2_name    : Argument to update the mobilenet version 2 detection model (if required)
        """
        
        self.m2_model = None
        self.mAlex_model = None
        self.vggf_model = None
        self.prediction_method = None
        self.label_dict = {0 : "Empty", 1 : "Occupied"}
        
        
        if model_name == "m2":
            self.load_mobilenet_v2(root_dir, m2_name)
            self.prediction_method = self.m2_prediction
            self.model_name = "Mobilenet V2"
        if model_name == "mAlex":
            self.load_miniAlex(root_dir)
            self.prediction_method = self.mAlex_prediction
            self.model_name = "Mini Alexnet"
        if model_name == "vgg-f":
            self.load_vggf(root_dir)
            self.prediction_method = self.vggf_bkbn_pass
            self.model_name = "VGG_F-SVM"
        
        
        
    def backbone_detection_all(self, input_image, extractor_obj):
        
         
        """ It performs the occupency detection on each individual parking spot by extracting pixels for individual parking spots using extractor object and calling respective prediction method.
        
        Params
        input_image   : Raw input image of the parking lot
        extractor_obj : Object of pixel extractor class containing information of individual parking lot
        
        Returns
        prediction_dict : Dictionay containing detection output of individual parking spot
        time            : time taken to finish entire detection (miliseconds)
        
        
        
        """
        prediction_dict = {}
        
        print("Backbone start")
        
        sp = time.time()
        spot_patch_list = []
        for ind, spot_data in enumerate(extractor_obj.spot_coordinate_list):
            spot_cord = spot_data[0]
            spot_patch_list.append(extractor_obj.crop_image_by_coordinates(input_image, spot_cord[:4]))
            
        final_pred_list, exec_time, pred_conf_list = self.prediction_method(spot_patch_list)
        
        for ind, item in enumerate(final_pred_list):
            prediction_dict[ind] =  {"prediction":item, "confidence":pred_conf_list[ind]}  
        
        print("Backbone end")
        
        return prediction_dict, time.time() - sp
        
        
        
#     MOBILENET V2    
    def load_mobilenet_v2(self, root_dir, m2_name):
        
        """
        It loads the tensorflow model of mobilenet version 2 into an object attribute
        
        root_dir : Root directory of the project
        m2_name  : Name of the target directory in model folder
        """
        
#         model_path = os.path.join(root_dir, "Models", "Mobilenet V2")
        model_path = os.path.join(root_dir, "Models", m2_name)
        
        print("Loading model")
        self.m2_model = tf.keras.models.load_model(model_path)
        print("Model Loaded")
       
        
    def m2_prediction(self, img_list):
        
        """
           It perfomrs the detection on each image provided in the argument using mobilenet version 2 model and returns the detection, time taken and confidence of prediction.
           
           Param
               img_list : List of individual parking spot
           
           Returns
               Detection output list
               Time taken by model to process all images,
               Confidence/probability of car being parked at individual parking spot
           
           
        """
        
        temp_list = []
        
        for img in img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            temp_list.append(cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA))
        sp = time.time()
        predict = self.m2_model.predict(np.array(temp_list))
        backbone_time = time.time() - sp
        prediction = tf.nn.sigmoid(predict)
        pred = np.where(prediction < 0.5, self.label_dict[0], self.label_dict[1])
        
        return pred[:,0].tolist() , backbone_time, prediction.numpy()[:,0].tolist()
    
    
#    MINI ALEXNET
    
    def load_miniAlex(self, root_dir):
        
        """
        It loads the caffe model of the mini Alexnet as a class attibute
        
        root_dir : Path to the root directory of the project
        
        """
        
        print("Loading model")
        mini_alex_net_caffe_path = os.path.join(root_dir, "Models", "Mini Alexnet")
        prototxt = os.path.join(mini_alex_net_caffe_path, "deploy.prototxt")
        caffe_model = os.path.join(mini_alex_net_caffe_path, "snapshot_iter_6318.caffemodel")
        mini_alexnet_model = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
        self.mAlex_model = mini_alexnet_model
        print("Model Loaded")
        
        
    def mAlex_prediction(self, img_list):
        
        """
        It perfomrs the detection on each image provided in the argument using mini Alexnet model and returns the detection, time taken and confidence of prediction.
        
        Param
               img_list : List of individual parking spot
           
           Returns
               Detection output list
               Time taken by model to process all images,
               Confidence/probability of car being parked at individual parking spot
        
        """

        blob = cv2.dnn.blobFromImages(img_list, 1. / 256, (224, 224), (0, 0, 0))
        self.mAlex_model.setInput(blob)
        stp = time.time()
        output = self.mAlex_model.forward() # miniALexnet return probability for both emty and occupancy [ 0.3 (empty), 0.7(occupied)]
        back_time = time.time() - stp
        occupency_prob = output[:,1]
        pred = np.where(np.array(occupency_prob) < 0.5, self.label_dict[0], self.label_dict[1])
        return pred.tolist(), back_time, occupency_prob.tolist()
    
#     VGG-F

    def load_vggf(self, root_dir):
        
        """
        It loads both parts of the VGG-F detection model
            tensorflow model which is a feature extractor
            vgg-f model that acts like final classifier
        """
        
        print("Loading Model")
        vgg_f_path = os.path.join(root_dir, "Models", "VGG-F","VGG-f")
        classifier_path = os.path.join(root_dir, "Models", "VGG-F","VGGF_SVM.joblib")
        self.input_shape = (224, 224, 3)                                                 
        self.vggf_model = tf.keras.models.load_model(vgg_f_path)                         # Feature extractor model
        self.svm = joblib.load(classifier_path)                                          # Final predictor SVM
        print("Model Loaded")
        

    def vggf_bkbn_pass(self, img_list):
        
         """
        It perfomrs the detection on each image provided in the argument using VGGF-SVM based detection model and returns the detection, time taken and confidence of prediction.
        
        Param
               img_list : List of individual parking spot
           
           Returns
               Detection output list
               Time taken by model to process all images,
               Confidence/probability of car being parked at individual parking spot (Dummy variable to maintain consistency with other models output for dynamic use of the method)
        
        """
        
        temp_list = []
        prob_list = []         # Dummy variable for probability to maintain consistency with other models detection methods
        for img in img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            temp_list.append(cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA))
            prob_list.append(np.nan)
            
        sp = time.time()
        inp_img_list = np.array(temp_list)
        feature_list = self.vggf_model.predict(inp_img_list)      # Feature extraction
        prediction_list = self.svm.predict(feature_list)          # Final prediction
        fp = time.time()
        total_time = fp - sp
        return prediction_list.tolist(), total_time, prob_list
        
        
        