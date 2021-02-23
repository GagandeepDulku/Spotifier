from Helper.helper import *

class Processor():
    
    def __init__(self, spot_dict):
        
        self.spot_dict = spot_dict
        self.clear_index_list = [0,3] # (overlap result with these index require no occlusion handling) 0 means there is 0 overlap and 3 means the overlap is bevause of parked car
        self.last_pred_list = {}
        self.initiate_cache()
        
        
    def initiate_cache(self):
        default_pred = "Empty"
        default_conf = 0.0
        
        for spot_name in list(self.spot_dict.keys()):
            if self.spot_dict[spot_name][4] == True:
                self.last_pred_list[spot_name] = {"prediction":"Occupied",
                                                   "confidence":1}
            else:
                self.last_pred_list[spot_name] = {"prediction":"Empty",
                                                   "confidence":0}
            
            
    def forward_pass(self, occ_list, backbone_list, use_occ_result = True):
        
        """
        This method processes detection model's output and occlusion locator ooutput to produce final detection output
        Params
            occ_list       : Output of occlusion locator
            backbone_list  : Output of detection model
            use_occ_result : Enable/disable the effect of occlusion locator 

        Returns
            final_prediction_dict : dictionay containing final detection output
        
        """
        
        final_prediction_dict = {}
         
        for spot_ind, spot_name in enumerate(list(self.spot_dict.keys())):
            cord, min_overlap, max_overlap, overflow_thres,_ = self.spot_dict[spot_name]
            spot_cord = (cord[0], cord[2])
            occ_proc_dict = self.get_occlusion_status(spot_cord, occ_list, min_overlap, max_overlap, overflow_thres)
            
            if use_occ_result and occ_proc_dict["occlusion_type_ind"] not in self.clear_index_list:
                pred = self.last_pred_list[spot_name]["prediction"]
                conf = self.last_pred_list[spot_name]["confidence"]
            else:
                pred = backbone_list[spot_ind]["prediction"]
                conf = backbone_list[spot_ind]["confidence"]
                
                self.last_pred_list[spot_name]["prediction"] = pred
                self.last_pred_list[spot_name]["confidence"] = conf
                 
            final_prediction_dict[spot_name] = {"prediction":pred, 
                                               "confidence":conf, 
                                               "occlusion_status":occ_proc_dict,
                                               "backbone_result":backbone_list[spot_ind]}
            
        return final_prediction_dict
    
    def get_occlusion_status(self, spot_cord, occlusion_list, min_thres, max_thres, occ_overflow_thres):
        
        """
        It evaluates any potential occlusion by mathcing overlapping between pixel coordinates of parking spot and detected objects
        
        Params:
            spot_cord          : Pixel coordinates of the parking spot
            occlusion_list     : List of objects detected by occlusion locator, it includes pixel coordinates
            min_thres          : Minimum threshold of overlap
            max_thres          : Maximum threshold of overlap
            occ_overflow_thres : Threshold of overflow

        Returns:
            occ_dict : Output of occlusion related conflict represented as dictionary
        
        """
        
        occ_dict = {"occlusion_type_ind" : 0} # default No occlusion
        for ind, it in enumerate(occlusion_list):
            occ_cord = it[2]
            overlap_pixel_count = overlap_bbox(occ_cord, spot_cord)
            spot_pixel_count = get_pixel_area(spot_cord[0], spot_cord[1])
            occ_obj_pixel_count = get_pixel_area(occ_cord[0], occ_cord[1])

            overlap_share = overlap_pixel_count/spot_pixel_count
            if overlap_pixel_count > 0:
                occ_overflow = (occ_obj_pixel_count - overlap_pixel_count)/occ_obj_pixel_count
            else:
                occ_overflow = 0
#             print(occ_obj_pixel_count)
#             print(overlap_pixel_count)
#             print(occ_overflow)
            
            if overlap_share <= min_thres:
                if occ_dict["occlusion_type_ind"] != 3:
                    occ_dict["occlusion_type_ind"] = 0
                    occ_dict["occlusion_type"] = "No Occlusion"
                    occ_dict["overlap"] = overlap_share
                    occ_dict["overflow"] = occ_overflow
                    occ_dict["coordinates"] =  "None"
                    occ_dict["label"] = "None"
            elif overlap_share < max_thres:
                occ_dict["occlusion_type_ind"] = 1
                occ_dict["occlusion_type"] = "Partial Occlusion"
                occ_dict["overlap"] = overlap_share
                occ_dict["overflow"] = occ_overflow
                occ_dict["coordinates"] =  occ_cord
                occ_dict["label"] = it
                break
            elif occ_overflow > occ_overflow_thres:
                occ_dict["occlusion_type_ind"] = 2
                occ_dict["occlusion_type"] = "Complete Occlusion"
                occ_dict["overlap"] = overlap_share
                occ_dict["overflow"] = occ_overflow
                occ_dict["coordinates"] =  occ_cord
                occ_dict["label"] = it
                break
            else:
                occ_dict["occlusion_type_ind"] = 3
                occ_dict["occlusion_type"] = "Occupied"
                occ_dict["overlap"] = overlap_share
                occ_dict["overflow"] = occ_overflow
                occ_dict["coordinates"] =  occ_cord
                occ_dict["label"] = it
        
        return occ_dict