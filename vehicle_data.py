import numpy as np
import os


test_size = 0.3
random_state = 2019

train_list_file = "/Users/batele/Desktop/DeepLearning/data/VehicleID_V1.0/train_test_split/train_list.txt"
model_attr_file = "/Users/batele/Desktop/DeepLearning/data/VehicleID_V1.0/attribute/model_attr.txt"

train_valid_save_path = "/Users/batele/Desktop/DeepLearning/method1/train_valid/"

img_base_path = "/Users/batele/Desktop/DeepLearning/data/VehicleID_V1.0/image/"

def vehicleid_imgnames(file):
    """
        train_list: <车图片地址> <车id>
        
        return: <车id>: [车图片地址s]
    """
    vid2img_dict = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            
            if line.count('\n') != len(line):
            
                img_vid = line.split(' ')
                img = img_vid[0]
                vid = img_vid[1].replace('\n','')
                value = [img]
                if vid in vid2img_dict.keys():
                   vid2img_dict[vid].append(img)
                else:
                    vid2img_dict[vid] = value

    return vid2img_dict

def vid_model(file):
    """
        model_attr: <车辆id> <车模型>
        return: <车辆id> <车模型>
    """
    vid2model_dict = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            
            if line.count('\n') != len(line):
                
                vid_model = line.split(' ')
                vid = vid_model[0]
                model = vid_model[1].replace('\n','')
                vid2model_dict[vid] = model

    return vid2model_dict

def train_valid_data():

    vid2model_dict = vid_model(model_attr_file)
    vid2imgs_dict = vehicleid_imgnames(train_list_file)

    
    imgid2model_list = []
    
    for key in vid2imgs_dict.keys():
    
        if key in vid2model_dict.keys():
            model = vid2model_dict[key]

            for imgname in vid2imgs_dict[key]:
                image_full_path = img_base_path + imgname + ".jpg"
                image_model_string = image_full_path + " " + model + "\r\n"
                imgid2model_list.append(image_model_string)

    np.random.shuffle(imgid2model_list)
    index = int(len(imgid2model_list) * (1-test_size))
    
    train_imgid2model_list = imgid2model_list[:index]
    valid_imgid2model_list = imgid2model_list[index:]
    
    train_full_path = train_valid_save_path + "train_vid_model.txt"
    valie_full_path = train_valid_save_path + "valid_vid_model.txt"
    
    with open(train_full_path,'w') as f:
    
        for i in train_imgid2model_list:
            f.write(i)
    
    with open(valie_full_path,'w') as f:
    
        for i in valid_imgid2model_list:
            f.write(i)

    return
if __name__ == "__main__":

    train_valid_data()

