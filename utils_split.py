import os
import json
import pickle
import random

import matplotlib.pyplot as plt


def split_Medlat(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    str_type = 'bands'
    print("using:",str_type)
    
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla)) and cla.split('_')[-1]==str_type]
    
    class_indices = {}
    for dir in flower_class:
        if dir.split('_')[0]== 'pulsars':
            class_indices[dir] = 1
        else:
            class_indices[dir] =0

    
    train_images_path = []  
    train_images_label = []  
    val_images_path = [] 
    val_images_label = [] 
    test_images_path = [] 
    test_images_label =[] 
    every_class_num = [] 

    
    val_rfi_numbers = [867]*8
    val_rfi_numbers.append(867)

    
    test_rfi_numbers = [1328] * 8 
    test_rfi_numbers.append(1336)
    i=0
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
    
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        
        image_class = class_indices[cla]
        
        every_class_num.append(len(images))
        
        if image_class == 1:
            #test_path = random.sample(images, k=500)
            test_path = random.sample(images, k=int(len(images) * 1))
            #val_path = random.sample(images, k=500)
        else:
            val_path = random.sample(images,k=val_rfi_numbers[i])

            test_path = random.sample(del_list_elements(images,val_path), k=test_rfi_numbers[i])
            i+=1

            #val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in test_path:  
                test_images_path.append(img_path)
                test_images_label.append(image_class)

                    
            elif img_path in val_path: 
                val_images_path.append(img_path)
                val_images_label.append(image_class)

            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label,test_images_path,test_images_label


def split_PMPS(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    str_type = 'bands'
    print("using:",str_type)
    
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla)) and cla.split('_')[-1]==str_type]
   
    class_indices = {}
    for dir in flower_class:
        if dir.split('_')[0]== 'pulsar':
            class_indices[dir] = 1
        else:
            class_indices[dir] =0

    
    train_images_path = []  
    train_images_label = [] 
    val_images_path = []  
    val_images_label = []  
    test_images_path = [] 
    test_images_label =[] 
    every_class_num = []  

  
    val_rfi_numbers = [100,100]
    
    test_rfi_numbers = [200,9800] 
    
    i=0
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
    val_path  = []
    
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        
        image_class = class_indices[cla]
        
        every_class_num.append(len(images))
        
        
        if image_class == 1: 
            
            test_path = random.sample(images, k=int(len(images) * 1))
           
        else:#
            val_path = random.sample(images,k=val_rfi_numbers[i])

            test_path = random.sample(del_list_elements(images,val_path), k=test_rfi_numbers[i])
            i+=1

            
            
        for img_path in images:
            if img_path in test_path:  
                test_images_path.append(img_path)
                test_images_label.append(image_class)

                    
            elif img_path in val_path: 
                val_images_path.append(img_path)
                val_images_label.append(image_class)

            elif image_class!=1:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    

    return train_images_path, train_images_label, val_images_path, val_images_label,test_images_path,test_images_label



def del_list_elements(in_list, del_list):
    
    list_keys = [i for i in range(len(in_list))]
    temp_dict = dict(zip(list_keys, in_list))
    
    temp_list = []
    for item in del_list:
        [temp_list.append(k) for k, v in temp_dict.items() if v == item]
    
    [temp_dict.pop(i) for i in temp_list]
    
    new_temp_list = list(temp_dict.values())
    return new_temp_list




def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
if __name__ =="__main__":
    root = '../Data/BNUDataset_48x48'
    split_bnudataset(root)
