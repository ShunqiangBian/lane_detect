import os
import pandas as pd
from sklearn.utils import shuffle
from utils.logconfig import getLogger

"""
存放image和label路径地址的list
note：需要使用绝对路劲，防止出现 NoneTpye Object 的 Error
使用logger格式化训练日志
"""
label_list = []
image_list = []
image_dir = 'D:\\data\\LaneSeg\\Image_Data'
label_dir = 'D:\\data\\LaneSeg\\Gray_Label'
logger=getLogger()
"""
目录结构：
   Image_Data/
     road02/
       Record002/
         Camera 5/
           ...
         Camera 6
       Record003
       ....
     road03
     road04
   Gray_Label/
     Label_road02/
      Label
       Record002/
         Camera 5/
          ...
         Camera 6
       Record003
       ....
     Label_road03
     Label_road04     
     
"""

"""
Image_Data
以image路径为循环迭代遍历所有数据

image_sub_dir1 : image_dir/road02
label_sub_dir1 : label_dir/label_road02/label
image_sub_dir2 : image_dir/road02/record001/
label_sub_dir2 : label_dir/label_road02/label/record001
image_sub_dir3 : image_dir/road02/record001/camera 5
label_sub_dir3 : label_dir/label_road02/label/record001/camera 5
image_sub_dir4 : image_dir/road02/record001/camera 5/
label_sub_dir4 : label_dir/label_road02/label/record001/camera 5/

Dataset & Training 数据分割
total_dataset：100%
train_dataset：60%
val_dataset：20%
test_dataset：20%
"""
logger.info("Make list start...")
for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')
    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)
        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg','_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                if not os.path.exists(image_sub_dir4):
                    logger.warn("{}".format(image_sub_dir4))
                    continue
                if not os.path.exists(label_sub_dir4):
                    logger.warn("{}".format(label_sub_dir4))
                    continue
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)

assert len(image_list) == len(label_list)
logger.info("The length of image dataset is {}, and label is {}".format(len(image_list), len(label_list)))
total_length = len(image_list)
sixth_part = int(total_length*0.6)
eighth_part = int(total_length*0.8)
all = pd.DataFrame({'image':image_list, 'label':label_list})
all_shuffle = shuffle(all)
train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eighth_part]
test_dataset = all_shuffle[eighth_part:]
train_dataset.to_csv('../data_list/train.csv', index=False)
val_dataset.to_csv('../data_list/val.csv', index=False)
test_dataset.to_csv('../data_list/test.csv', index=False)
logger.info("Make list end...")
