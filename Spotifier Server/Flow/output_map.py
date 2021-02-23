import os
import cv2
import pickle

class Output_Map():
    
    def __init__(self, root_directory):
        self.spot_width = 50
        self.spot_height = 75
        os.path.join(root_directory, "Resources", "Images", "Occupied.png")
        self.occupied_icon = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(root_directory, "Resources", "Images", "Occupied.png")), (self.spot_width,self.spot_height), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.empty_icon = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(root_directory, "Resources", "Images", "Empty.png")), (self.spot_width,self.spot_height), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.map_coord_dict = self.read_list(os.path.join(root_directory, "Resources","ParkingLot","parking_lot_spot.dict"))
        
        


    def print_result_on_map(self, result):
        
        prediction = result["Prediction Dict"]   # Extrating detection output
        self.base = cv2.imread("base.png")       # Setting up the base image

        for k in self.map_coord_dict.keys():
            pos = self.map_coord_dict[k]
            status = prediction[k]["prediction"]
            if status == "Occupied":
                icon = self.occupied_icon
            else:
                icon = self.empty_icon

            if k[0] in ["b","d"]: # Flip certain rows image to make it look intiutive
                self.base[pos[1]:pos[1] +self.spot_height, pos[0]:pos[0] +self.spot_width,:] = cv2.flip(icon, 0)
            else:
                self.base[pos[1]:pos[1] +self.spot_height, pos[0]:pos[0] +self.spot_width,:] = icon
        return self.base
    
    
    def read_list(self, path):
        with open (path, 'rb') as fp:
            itemlist = pickle.load(fp)
        return itemlist