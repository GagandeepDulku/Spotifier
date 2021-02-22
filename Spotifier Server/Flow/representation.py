import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np

class Representation():
    
    def __init__(self):
        
        
        self.occupied_rep = (0, 0, 255)
        self.vacant_rep = (0, 255, 154)
        
#         self.spot_list = spot_list
        
#         IMAGE_DIR = os.path.join(ROOT_DIR, "Resources", "Images")
        
        
#         self.empty_img = plt.imread(os.path.join(IMAGE_DIR, "Empty.png"))
#         self.occupied_img = plt.imread(os.path.join(IMAGE_DIR, "Occupied.png"))
        
        
        self.representation_dict = {"Empty":self.vacant_rep,
                                    "Occupied": self.occupied_rep}
        
        
        
    def get_representation(self, label):
        return self.representation_dict[label]
        
        
        
    def get_disp_pos(self, index, coord, hor_adj = 0, vert_adj = 0):
        
        "Create relative coordinates from input (coord), it is used for diplaying prediction data on video frame"

        vert_diff = 20
        top_left_x = coord[0]
        top_left_y = coord[1]

        return (top_left_x + hor_adj, top_left_y + vert_adj + vert_diff*index)
    
        
        
        
        
    def draw_result(self, frame, controller_obj, counter, count_flag, fract_flag, all_occ_flag, ol_occ_flag, change_list = None):
        
        """ Print prediction results on the image frame
            frame        : Input image (canvas to draw results)
            counter      : Represent index number of the frame
            count_flag   : It dictates weather to show no of time the spot was marked occupied and empty
            fract_flag   : It dictates weather to show confidence(probability), Occlusion share and Overlap share fractions
            all_occ_flag : It dictates weather all detected object shall be printed on the image
            ol_occ_flag  : It dictates weather the objects overlapping with parking spot shall be printed or not
            changlist    :    Represent list of status change of each spot    
        
        """
        font_size = 0.8
        font_thickness = 2
        font_type = cv2.FONT_HERSHEY_SIMPLEX
                
        for ind, spot_name in enumerate(list(controller_obj.extracter_obj.spot_dict.keys())):        # for each parking spots
            
            
        # Extract related info from prediction
        
            # Pixel coordinates of the spot
            coord = controller_obj.extracter_obj.spot_dict[spot_name][0]
            
            # Occupency status of the spot
            pred = controller_obj.final_prediction["Prediction Dict"][spot_name]["prediction"]
            
            # Probability of presence of car
            conf = controller_obj.final_prediction["Prediction Dict"][spot_name]["confidence"]       
            
        # Occlusion related data   
            occ_type_name = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["occlusion_type"]
            occ_overlap = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["overlap"]
            occ_overflow = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["overflow"]
            occ_cord = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["label"][2]
            occ_name = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["label"][0]
                
        # Execution time related data saved ad meta data
            controller_obj.bck_time_list.append(controller_obj.backbone_prediction_time)      
            controller_obj.occ_time_list.append(controller_obj.occ_loc_obj.detection_time)
            controller_obj.total_time_list.append(controller_obj.total_forward_time)
            

        # Update Execution Variable
            if pred == "Empty":
                controller_obj.track_status_list[ind].append(0)
            else:
                controller_obj.track_status_list[ind].append(1)   
                
            occup_count = controller_obj.track_status_list[ind].count(1)
            empty_count = controller_obj.track_status_list[ind].count(0)
            if(change_list != None):
                if len(controller_obj.track_status_list[ind]) > 1:
                    if(controller_obj.track_status_list[ind][-1] != controller_obj.track_status_list[ind][-2]):
                        change_list[ind] = change_list[ind]+1

        # Get representation (colour) of the prediction
            output_rep = self.get_representation(pred)

        # Draw status on Frame
            frame = cv2.rectangle(frame, coord[0], coord[2], output_rep, 2)
            frame = cv2.putText(frame, str(spot_name),
                                controller_obj.get_disp_pos(
                                0,
                                coord[0], vert_adj = -2),
                                font_type,
                                fontScale= font_size, 
                                color=(255,0,0),
                                thickness = font_thickness)
            
            
            
            if(change_list != None) and (count_flag):
                
            # Print no of times parking spot was empty during execution 
                frame = cv2.putText(frame,
                                    str(empty_count),
                                    controller_obj.get_disp_pos(1, coord[0]),
                                    font_type,
                                    fontScale = font_size,
                                    color=(0,255,0),
                                    thickness = font_thickness)
                
            # Print no of times parking spot was occupied during execution 
                frame = cv2.putText(frame,
                                    str(occup_count),
                                    controller_obj.get_disp_pos(2, coord[0]),
                                    font_type,
                                    fontScale = font_size,
                                    color = (0,0,255),
                                    thickness = font_thickness)
                
                
        # Print meta data from detection model and occlusion locator in terms of fractions     
            if fract_flag:
                
            # Probability/confidence of the spot being occupied
                frame = cv2.putText(frame,
                                    "cnf:"+str(round(conf, 3)),
                                    controller_obj.get_disp_pos(4, coord[0]),
                                    font_type,
                                    fontScale= font_size,
                                    color=(0,255,255),
                                    thickness = font_thickness)
                
            # Value of overlapping fraction for the spot
                frame = cv2.putText(frame,
                                    "OL:"+str(round(occ_overlap,3)),
                                    controller_obj.get_disp_pos(5, coord[0]),
                                    font_type,
                                    fontScale = font_size,
                                    color = (0,255,255),
                                    thickness = font_thickness)
                
            # Value of the overflow fraction for the spot
                frame = cv2.putText(frame,
                                    "OF:"+str(round(occ_overflow,3)),
                                    controller_obj.get_disp_pos(6, coord[0]),
                                    font_type,
                                    fontScale = font_size,
                                    color = (0,255,255),
                                    thickness = font_thickness)
                
            
        # Print the detected occlusion that has overlap with the spot
            if((controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["occlusion_type_ind"] in [1,2]) and ol_occ_flag):
                occ_cord = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["label"][2]
                occ_name = controller_obj.final_prediction["Prediction Dict"][spot_name]["occlusion_status"]["label"][0]
                frame = cv2.rectangle(frame, occ_cord[0], occ_cord[1], (255, 255,0), 2)
                frame = cv2.putText(frame,
                                    occ_name,
                                    (occ_cord[1][0], occ_cord[1][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale= 0.7,
                                    color=(255,255,0),
                                    thickness = 2)
                
         # Print all objeccts detected by occlusion locator       
            if all_occ_flag:
                for occ_cord_data in controller_obj.occ_loc_obj.result_list:
                    occ_cord = occ_cord_data[2]
                    occ_name = occ_cord_data[0]
                    occ_conf = str(round(occ_cord_data[1],2))
                    frame = cv2.rectangle(frame, occ_cord[0], occ_cord[1], (255, 255,0), 2)
                    frame = cv2.putText(frame,
                                        occ_name +" : "+occ_conf,
                                        (occ_cord[1][0], occ_cord[1][1]),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale= 0.7,
                                        color=(255,255,0),
                                        thickness = 2)
                    
                    
     # Generic prints on top left corner             
        # Print the frame count            
        frame = cv2.putText(frame, 
                            str(int(counter)),
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 1,
                            color=(0,0,0),
                            thickness = 3)
        
        # Print the backbone time
        frame = cv2.putText(frame,
                            str(round(np.mean(controller_obj.backbone_prediction_time),4)),
                            (30, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 1,
                            color=(0,0,0),
                            thickness = 3)
        # Print the time taken by occlusion locator
        frame = cv2.putText(frame,
                            str(round(np.mean(controller_obj.occ_loc_obj.detection_time),4)),
                            (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 1,
                            color=(0,0,0),
                            thickness = 3)
        
        # Print the total time taken to complete the detection
        frame = cv2.putText(frame,
                            str(round(np.mean(controller_obj.total_forward_time),4)),
                            (30, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 1, color=(0,0,0), thickness = 3)
                    
        return frame
        
