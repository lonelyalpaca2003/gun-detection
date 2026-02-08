import cv2
import os



def convert_to_yolo(x_min:int, x_max:int, y_min:int, y_max:int, w:int, h:int):
    """Converts the image files (currently in absolute pixel values and the format x_min, y_min, x_max, y_max) into
        a format that is needed to train YOLO models. This function converts it into values between 0-1 and into the format 
        x_center, y_center, width and height"""
    x_center = (x_min + x_max)/(2 * w)
    y_center = (y_min + y_max)/(2 * h)
    width = (x_max - x_min)/w
    height = (y_max - y_min)/h
    return x_center,  y_center, width, height



def create_splits(current_img_path: str, destination_path : str, current_label_path : str, 
                  label_destination : str,  split_size : float, train : bool = True):
        
        image_files = sorted(os.listdir(current_img_path))
        label_files = sorted(os.listdir(current_label_path))
        split_idx = int(len(os.listdir(current_img_path)) * 0.8)
        if train: 
            sample = image_files[:split_idx]
            label = label_files[:split_idx]
        else:
            sample = image_files[split_idx:]
            label = label_files[split_idx:]
            print(len(sample))    
        for file in sample: 
            image_path = os.path.join(current_img_path, file)
            image = cv2.imread(image_path)
            h, w, _ = image.shape
            cv2.imwrite(os.path.join(destination_path, file), image)

            label_paths = os.path.join(current_label_path, file[:-4] + "txt")
            f = open(label_paths, "r")
            lines = f.readlines()
            f.close()
            
            f = open(os.path.join(label_destination, file[:-4] + "txt"), "w")

            for line in lines[1: ]:
                    line = line.strip().split()
                    x_min, y_min, x_max, y_max = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                    x_center, y_center, width, height = convert_to_yolo(x_min, x_max, y_min, y_max, w, h)
                    f.write("{} {} {} {} {}\n".format(0, x_center, y_center, width, height))