import cv2
import numpy as np
from xml.dom import minidom

class PixelExtracter():
    
    def __init__(self, xml_path, default_thres_list, special_spot, parked_spots):
        
        self.xml_file_path = xml_path
        self.default_thres_list = default_thres_list
        self.special_spot = special_spot
        self.parked_spots = parked_spots
        self.spot_dict = {}
        self.spot_coordinate_list = self.seek_spot_lists(self.xml_file_path)
        
    def seek_spot_lists(self, xml_path):
        
        """
        It reads the pixel coordinates from the xml file and create a list of pixel coordinats of all parking spots mentioned in the xml file
        Params    
            xml_path  : Path to the xml file
        Returns
            spot_list : list of tuples containing pixel coordinates of parking spots
        """
        
        spot_dict = self.get_pixel_cordinate_from_xml(xml_path)
        self.spot_dict = spot_dict
        spot_list = list(spot_dict.values())
        return spot_list
    
    def get_pixel_cordinate_from_xml(self, path):
        
        """
        It reads the xml file and process the data and create a dictionary representation of all pixel coordinates for respective parking spots
        Params
            path      : Path to the xml file
        Returns
            spot_dict : Dictionary containing all pixel coordinates for respective spots
            
        """
        
        
        spot_dict = {}
        try:
            xml_doc = minidom.parse(path)
            for space_list in xml_doc.getElementsByTagName("object"):
                spot_name = str(space_list.getElementsByTagName("name")[0].firstChild.nodeValue)
                min_overlap_thres = self.default_thres_list[0]
                max_overlap_thres = self.default_thres_list[1]
                overflow_thres = self.default_thres_list[2]
                if spot_name in list(self.special_spot.keys()):
                    for thres_name in self.special_spot[spot_name].keys():
                        if thres_name == "min_overlap_thres":
                            min_overlap_thres = self.special_spot[spot_name]["min_overlap_thres"]
                        if thres_name == "max_overlap_thres":
                            max_overlap_thres = self.special_spot[spot_name]["max_overlap_thres"]
                        if thres_name == "overflow_thres":
                            overflow_thres = self.special_spot[spot_name]["overflow_thres"]
                
                for spot in space_list.getElementsByTagName("bndbox"):
                    cordinate_list = []
                    cordinate_list.append(int(spot.getElementsByTagName("xmin")[0].firstChild.nodeValue))
                    cordinate_list.append(int(spot.getElementsByTagName("ymin")[0].firstChild.nodeValue))
                    cordinate_list.append(int(spot.getElementsByTagName("xmax")[0].firstChild.nodeValue))
                    cordinate_list.append(int(spot.getElementsByTagName("ymax")[0].firstChild.nodeValue))
                    
                    
                if spot_name in self.parked_spots:
                    init_status = True
                else:
                    init_status = False
                    
                spot_dict[spot_name] = [[(cordinate_list[0], cordinate_list[1]), (cordinate_list[0], cordinate_list[3]), (cordinate_list[2], cordinate_list[3]), (cordinate_list[2], cordinate_list[1])],min_overlap_thres, max_overlap_thres, overflow_thres, init_status]
        except FileNotFoundError as fnf:
            print("XML File not found at the location :"+str(path))
            print(fnf)
        except Exception as e:
            print(e)
        return spot_dict
    
    def get_patch_mean_dimensions(self, path_lits):
        temp_w = []
        temp_h = []
        for e_cord in  list(path_lits.values()):
            width = e_cord[2][0]  - e_cord[0][0]
            height = e_cord[1][1]  - e_cord[0][1]
            temp_w.append(width)
            temp_h.append(height)
        mean_width = int(np.mean(temp_w))
        mean_height = int(np.mean(temp_h))
        return mean_width, mean_height
        
        # look for xml_file
        
        
        
 #------------- LEGECY CODE USED FOR EARLIER XML FILES AND SET UP------------------------------       
        
    def read_spot_coordinates_from_xml(self, path, spot_tag="space", coordinate_tag="point", x_attrib="x", y_attrib="y", angle_tag="angle", angle_attrib="d"):
        """
        Read xml file and returns a list of parking spot's meta data as rows(list), each row = [(4 coordinates ,each is tuple), angle of tiltation]

        path : location of xml file
        spot_tag : tag name of individual spot in xml
        coordinate_tag : tag name of coordinates in xml 
        x_attrib : attribute name of x coordinate in xml
        y_attrib : attribute name of y coordinate in xml
        angle_tag : tag name of angle in xml
        angle_attrib : attribute name of angle in xml

        """
        space_cord_list = []

        try:
            xml_doc = minidom.parse(path)
            space_list = xml_doc.getElementsByTagName(spot_tag)
            for space in space_list:
                space_cord_list.append([(int(cord.attributes[x_attrib].value), int(cord.attributes[y_attrib].value)) for cord in space.getElementsByTagName(coordinate_tag)])
                space_cord_list[-1].extend([int(space.getElementsByTagName(angle_tag)[0].attributes[angle_attrib].value)])

        except FileNotFoundError as fnf:
            print("XML File not found at the location :"+str(path))
            print(fnf)
        except Exception as e:
            print(e)

        return space_cord_list
    
    def mask_img_region(self, image, coordinate_list):
        
        """
        This method is used to mask the region fo pixel given as pixel coordinates. It is used if selected region needed to be cropped out.
        """
        
        
      # Mask to locate cropping region (CR)
        mask = np.ones((image.shape),dtype=np.uint8)
        mask.fill(255)
        
      # Mark the coorinate region as black
        masked_image = cv2.fillPoly(mask, np.array([coordinate_list], dtype=np.int32),0)
        
      # Extract coorinate region from original image and rest is white 
        cropped = cv2.bitwise_or(image, masked_image)
        
        return cropped
    
    def rotate_and_wrap_image(self, image, degree_of_rotation):

        """ Rotate image at given angle/degree of rotation"""

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degree_of_rotation, 1.0)
     # borderMode (constant) and borderValue are important for maintaiing consistency    
        ri = cv2.warpAffine(image, rot_mat, image.shape[1::-1],  flags=cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT,borderValue = (255,255,255))
        return ri

    def crop_image_by_coordinates(self, input_img, coordinate_list):
        return input_img[coordinate_list[0][1]: coordinate_list[2][1], coordinate_list[0][0]: coordinate_list[2][0],:]
    
    # replace this function with above crop_image_by_coordinates
    def crop_and_rotate_image_by_coordinates_remove(self, image, coordinate_list, degree_of_rotation = None):
    
        """
        Crop certain region of image based on coordinates nad then rotate the cropped image for parallel view

        image : 3-D numpy array
        coordinate_list : list of tuples (top-left, bottom-left, bottom-right, top-right)
        degree_of_rotation = degree of rotation (if None, no rotation is performed)

        """
        print(coordinate_list)
        selected_region = self.mask_img_region(image, coordinate_list)
        if degree_of_rotation:
            selected_region = self.rotate_and_wrap_image(selected_region, degree_of_rotation)


    # remove the extra white and black pixels from the CR
        print(selected_region.shape)
        print(selected_region)
#         rows, col = np.where((selected_region[:,:,1] > 0)&(selected_region[:,:,1] < 255))
        new_img = selected_region[rows.min():rows.max(), col.min():col.max()]

        return new_img