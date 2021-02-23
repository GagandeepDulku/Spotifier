import os
import time
import sys
import yaml
from threading import Thread
from Flow.pixel_extracter import PixelExtracter
from Flow.representation import Representation
from Flow.backbone_detector import BackboneDetector
from Flow.occlusion_detector import OcclusionDetector
from Flow.processor import Processor
from Helper.helper import *

class Controller():
    
    def __init__(self, backbone_name, occ_detection_thres, primary_dir, track_dict_path = "track_dict.dict"):
        
        
        """ 
        CONSTRUCTOR
        
        backbone_name       : Name of the detection model, used for backbone detection instantiation
        occ_detection_thres : Minimum confidence for occlusion detection to pass as detected entity
        primary_dir         : Path of the root directory
        xml_path            : Path to the pri
        track_dict_path     : Path (including name) of the file where the tracked detectio data is to be saved
        
        """
        
     # Setting Root Directory
        sys.path.append(primary_dir)
        os.chdir(primary_dir)
        
        
     # Load Configurations
        config_file_path = os.path.join(primary_dir, "Configuration","config.yml")
        xml_path, default_thres, special_spot, parked_spots = self.load_config_file(config_file_path)
        
     # Reading parking spots pixel coordinates  
        self.xml_path = os.path.join(primary_dir, xml_path)
        print(self.xml_path)
        
        self.extracter_obj = PixelExtracter(self.xml_path, default_thres, special_spot, parked_spots)
        
     # Creating Backbone Detector objext  
        self.backbone_detector = BackboneDetector(backbone_name, primary_dir)
        
     # Creating Respresentation Obj (Used for Video Creation)
        self.rep_obj = Representation()
        
     # Creating Occlusion Dector Object
        self.occ_loc_obj = OcclusionDetector(occ_detection_thres)
        
     # Creating Processor Object
        self.proc_obj = Processor(self.extracter_obj.spot_dict)
        
     
        self.result_list = []                # List to contain all final predictions
        self.backbone_pred = {}              # Hold the value of all prediction from Backbone network (index = crop pos)
        
        self.bck_time_list = []              # List to keep track of backbone time per frame 
        self.occ_time_list = []              # List to keep track of Occlusion detection time per frame
        self.total_time_list = []            # List to keep track of Total time per frame
        
        
        self.backbone_prediction_time = None # Time to execute backbone network for all spots
        self.track_status_list = {}          # List to track occupancy status per frame for each spot
        self.conf_dict = {}                  # List to track confidence (probability) of the backbone for occpuancy per frame for each spot
        self.complete_track = {}             # List to track complete prediction result 
        
      # Initiating the tracking list
        for t in range(len(self.extracter_obj.spot_coordinate_list)):
            self.track_status_list[t] = []   # Used to display changed status
            self.complete_track[t] = []      # Saving the complete prediction list spot wise
            self.conf_dict[t] = []           # confidence list probably not needed remove it later
            
            
           
            
        self.track_dict_path = track_dict_path
        
      # Used for video update to reduce the strings n final video
        self.occ_type_map_dict = {"Complete Occlusion":"C.O.",
                                  "Partial Occlusion": "P.O.",
                                  "No Occlusion": "N.O.",
                                  "Occupied": "Occ-N.O."}
        
      # This method set up the input video stream (it can be from video or Camera)
        
        self.frame = None
        self.out = None
        
        
        
        
    def load_config_file(self, yml_path):
        
        """
        This function access the configuration file at given path and extract the configuration arguments explicitely provided
        
        yml_path : Path to the configuration file
        
        """
        
        with open(yml_path, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        xml_path = cfg["Coordinate_xml_path"]             # Path to file containing pixel coordinates
        default_thres = list(cfg["Default"].values())     # Default threshold values
        special_spot = cfg["Special_spot"]                # Parking spot that might require threshold values different from default
        parked_spot = cfg["Parked_spots"].split(" ")      # Name of the spots where car is parked when system is initiated
        
        return xml_path, default_thres, special_spot, parked_spot
     
    def set_up_input_stream(self, path_of_video):
        
        """" Set up video stream (It can be from Video or Camera )
             path_of_video : location of the input video
        """

        print(path_of_video)
        self.cap = cv2.VideoCapture(path_of_video)
        

    def controller_forward(self, frame, occ_hand_flag = True):
        
        """
            It recieve a raw image of the parking lot and output the detection result (along with occlusion handling) in the form of a python dictionary.
            
            frame         : Raw image of the parking lot
            occ_hand_flag : Enable or disable the effect of occlusion handling processs 
        """
        
     # Marking initial time
        sp = time.time()
        init_point = time.ctime()
        
     # Parallel thread for Occlusion detection
        frame_copy = frame.copy()
        occ_loc_thread = Thread(target= self.occ_loc_obj.detect_objects, args = (frame_copy,))
        occ_loc_thread.start()
        
     # Backbone detection (in main thread)     
        self.backbone_pred, self.backbone_prediction_time = self.backbone_detector.backbone_detection_all(frame, self.extracter_obj) 
        occ_loc_thread.join()                      # At this point occlusion thread will be completed or main thread will wait
        frame_copy = None                          # Vacanting memory space
        

     # Collecting the output of the Threads
        occ_list = self.occ_loc_obj.result_list
        bck_pred = self.backbone_pred
        
     # Processing for final decision 
        self.final_prediction["Prediction Dict"] = self.proc_obj.forward_pass(occ_list, bck_pred, occ_hand_flag)

     # Recording Time
        self.total_forward_time = time.time() - sp # Total Execution time
        end_point = time.ctime()                   # Computer time at the time of Detection completion
        
     # Final time related entries   
        self.final_prediction["Detection Initiation"] = init_point
        self.final_prediction["Detection  Completion"] = end_point
        self.final_prediction["Total Backbone Time"] = self.backbone_prediction_time
        self.final_prediction["Occlusion Locator Time"] = self.occ_loc_obj.detection_time
        self.final_prediction["Total Detection Time"] = self.total_forward_time 
        
        return self.final_prediction
        
    
        
    def get_frame(self):
        
        """ 
        This method will get image of complete parking lot from source (Can use video or Camera) 
        
        """
        ret, frame = self.cap.read()
        self.frame = frame
        return ret, frame
        
        
    def get_status(self, occ_handle_flag = True, save_frame_flag = True, draw_status_flag = False):
        
        """ 
        This method with read one image and get prediction for that image and create an image with results on it.
        
        occ_handle_flag  : Enable/Disable occlusion handling
        save_frame_flag  : If true saves the frame with/ without printed resulted as an image
        draw_status_flag : Draw/Paint the detection result along with metadata on the input image
        
        """
        
        self.final_prediction = {}
        self.final_prediction = {"Status" : "Detection Failed"}
        ret, frame = self.get_frame()
      #   Validate the succesfull capture of the image (ret)
    
        if ret:
          # The detection status is saved in "final_prediction" 
            _ = self.controller_forward(frame, occ_handle_flag)        
            self.final_prediction["Status"] = "Detection Successful"
            
            if draw_status_flag:
              # Update the frame with results   
                frame = self.draw_result(frame, counter = 0,count_flag = False, fract_flag = True ,all_occ_flag = all_occ_flag, ol_occ_flag = True)                                                               
            
            if save_frame_flag:
                cv2.imwrite("prediction.jpg", frame)
                
        else:
            print("No image captured")
            

        return frame
        
        
        
    def get_disp_pos(self, index, coord, hor_adj = 0, vert_adj = 0):
        
        """Create relative coordinates from input (coord), it is used for diplaying prediction data on video frame
           index    : Position of the meta data in terms of line number (0 is top)
           coord    : Pixel coordiates of the parking spot box
           hor_adj  : Horizontal position adjustment, increasing this will move the position towards right
           vert_adj : Vertical position adjustment, increasing this will move the postion toward bottom
        
        """

        vert_diff = 20        # Padding between two rows 
        top_left_x = coord[0]
        top_left_y = coord[1]

        return (top_left_x + hor_adj, top_left_y + vert_adj + vert_diff*index)
    
    
    
                                  
    def sample_run(self, input_video_path, videoname, vid_width, vid_height, saving_fps, occ_handle_flag, count_flag, fract_flag, all_occ_flag, ol_occ_flag):
        
        """
        Method performs the detection on each frame of the video and print the result on the image and save it in the video.
        
        videoname        : Name of the output video
        vid_width        : Width fo the output resolution
        vid_height       : Height of the output resolution
        saving_fps       : Frames per seconds of the output resoutution
        occ_handle_flag  : Enable or disable occlusion handling
        
        count_flag       : Flag to determine weather to print detection count of individual parking spots or not
        fract_flag       : Flag to determine weather to print confidence, overlap and overflow fractions of individual parking spots
        all_occ_flag     : Flag to determine weather to print all detected object by occlusion locator on the image
        ol_occ_flag      : Flag to determine weather to print the objects/occlusions overlaping with parking spots
        
        """
        self.out_vid_width =  vid_width
        self.out_vid_height = vid_height
        
    # Name of the video frame
        window_name = "Parking"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        
    # Window property to full screen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    # Video Recording part
        self.out = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'MP4V'), saving_fps, (self.out_vid_width, self.out_vid_height))
        
    # The input video stream
        self.set_up_input_stream(input_video_path)
    
    # Execution Variables
    
       # Frame counter of the video (Displayed at top)
        counter_var = 0    
        
       # List to keep count of no of times occupancy changed for each (index of list) parking spot
        change_list = [0] * len(self.extracter_obj.spot_coordinate_list)  
      
    
        while(self.cap.isOpened()):
            counter_var = counter_var + 1

            frame = self.get_status(occ_handle_flag = occ_handle_flag, save_frame_flag = True, draw_status_flag = False)
            final_prediction = self.final_prediction
            if self.final_prediction["Status"] == "Detection Successful":
                self.result_list.append(self.final_prediction)
                write_list(self.track_dict_path, self.result_list)
                
                frame = self.rep_obj.draw_result(frame, self, counter_var, count_flag, fract_flag, all_occ_flag, ol_occ_flag, change_list)
                
                cv2.imshow(window_name,frame)
                frame = cv2.resize(frame, (self.out_vid_width, self.out_vid_height), interpolation = cv2.INTER_AREA)
                self.out.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                break
                    
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()