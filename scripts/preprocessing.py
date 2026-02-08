import os
from preprocess import create_splits

base_path = '../data'
image_path = base_path + '/Images'
label_path = base_path + '/Labels'

data_path = base_path + '/data.yaml'

train_img_path = '../data/train/images'
train_label_path = '../data/train/labels'
val_img_path = '../data/val/images'
val_label_path = '../data/val/labels'

os.makedirs(train_img_path, exist_ok = True)
os.makedirs(train_label_path, exist_ok = True)
os.makedirs(val_img_path, exist_ok= True)
os.makedirs(val_label_path, exist_ok= True)

create_splits(image_path, train_img_path, label_path, train_label_path, split_size = 0.8, train = True)
create_splits(image_path, val_img_path, label_path, val_label_path, split_size = 0.2, train = False)