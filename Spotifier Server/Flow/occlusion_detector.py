from darkflow.net.build import TFNet
import os
import time
import cv2
class OcclusionDetector():
    
    def __init__(self, detection_thres):
        self.darkflow_path = os.path.abspath("Flow/darkflow") 
        self.detector = None
        self.detection_thres = detection_thres
        self.result_list = []
        self.detection_time = None
        self.load_yolo_detector()
        
        self.excepted_labels = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
        
        
    def load_yolo_detector(self):
        current_path = os.path.abspath("")
        print(self.darkflow_path)
        os.chdir(self.darkflow_path)
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": self.detection_thres}
        self.detector = TFNet(options)
        os.chdir(current_path)
        
    def detect_objects(self, img):
        print("occ start")
        sp = time.time()
        self.result_dict = self.detector.return_predict(img)
        temp_list = []
        for occ_obj in self.result_dict:
            temp_list.append([occ_obj["label"] , occ_obj["confidence"], [(occ_obj["topleft"]["x"], occ_obj["topleft"]["y"]), (occ_obj["bottomright"]["x"], occ_obj["bottomright"]["y"])]])
        self.result_list = temp_list                          
#         self.result_list = [[occ_obj["label"] , occ_obj["confidence"], [(occ_obj["topleft"]["x"], occ_obj["topleft"]["y"]), (occ_obj["bottomright"]["x"], occ_obj["bottomright"]["y"])]] for occ_obj in self.result_dict]
        self.detection_time = time.time() - sp
        print("occ end")
        return self.result_list, self.detection_time
    
    def mark_object(self, img, colour = (255,0,0)):
        for i, it in enumerate(self.result_list):
            img = cv2.putText(img, str(i), (it[2][0][0] - 2, it[2][0][1] - 2),  cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.9, color=colour, thickness = 3)
            img = cv2.rectangle(img, it[2][0] , it[2][1], (255,0,0), 2)
        return img