import os
import json
import pickle
import random

import matplotlib.pyplot as plt


def split_Medlat(root: str, val_rate: float = 0.2, test_rate =1196/89996 ):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    str_type = 'bands'
    print("using:",str_type)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla)) and cla.split('_')[-1]==str_type]
    # # 排序，保证顺序一致
    # flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = {}
    for dir in flower_class:
        if dir.split('_')[0]== 'pulsars':
            class_indices[dir] = 1
        else:
            class_indices[dir] =0

    #class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = [] #存储测试集的所有图片路径
    test_images_label =[] #存储测试集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    # 用于每个文件夹下选取非脉冲星的个数
    #89996-1196=88800，88800*0.2=17760，17760=1973*8+1976
    val_rfi_numbers = [867]*8
    val_rfi_numbers.append(867)
#     val_rfi_numbers = [1200]*8 #675
#     val_rfi_numbers.append(676)
    
    
    test_rfi_numbers = [132] * 8 #132,55,2657、1328、664
    test_rfi_numbers.append(140)#140,60,2664、1336、668
    i=0
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        if image_class == 1: #脉冲星全部用于测试集
            #test_path = random.sample(images, k=500)
            test_path = random.sample(images, k=int(len(images) * 1))
            #val_path = random.sample(images, k=500)
        else:#
            val_path = random.sample(images,k=val_rfi_numbers[i])
#             val_path = random.sample(images,k=1)
            test_path = random.sample(del_list_elements(images,val_path), k=test_rfi_numbers[i])
            i+=1

            #val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in test_path:  # 如果该路径在采样的测试集样本中则存入测试验证集
                test_images_path.append(img_path)
                test_images_label.append(image_class)
#                 if image_class !=1:
#                     train_images_path.append(img_path)
#                     train_images_label.append(image_class)
                    
            elif img_path in val_path: # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
#                 train_images_path.append(img_path)
#                 train_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label,test_images_path,test_images_label


def split_PMPS(root: str, val_rate: float = 0.2, test_rate =1196/89996 ):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    str_type = 'bands'
    print("using:",str_type)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla)) and cla.split('_')[-1]==str_type]
    # # 排序，保证顺序一致
    # flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = {}
    for dir in flower_class:
        if dir.split('_')[0]== 'pulsar':
            class_indices[dir] = 1
        else:
            class_indices[dir] =0

    #class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = [] #存储测试集的所有图片路径
    test_images_label =[] #存储测试集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    # 用于每个文件夹下选取非脉冲星的个数
    #89996-1196=88800，88800*0.2=17760，17760=1973*8+1976
#     val_rfi_numbers = [1973]*8
#     val_rfi_numbers.append(1976)
    val_rfi_numbers = [1,1]
    
    test_rfi_numbers = [200,1800] 
    
    i=0
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    val_path  = []
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        
        # 按比例随机采样验证样本
        if image_class == 1: #脉冲星全部用于测试集
            #test_path = random.sample(images, k=500)
            test_path = random.sample(images, k=int(len(images) * 1))
            #val_path = random.sample(images, k=500)
        else:#
            val_path = random.sample(images,k=val_rfi_numbers[i])
#             val_path = random.sample(images,k=1)
            test_path = random.sample(del_list_elements(images,val_path), k=test_rfi_numbers[i])
            i+=1

            #val_path = random.sample(images, k=int(len(images) * val_rate))
            
        for img_path in images:
            if img_path in test_path:  # 如果该路径在采样的测试集样本中则存入测试验证集
                test_images_path.append(img_path)
                test_images_label.append(image_class)
#                 if image_class !=1:
#                     train_images_path.append(img_path)
#                     train_images_label.append(image_class)
                    
            elif img_path in val_path: # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
#                 train_images_path.append(img_path)
#                 train_images_label.append(image_class)
            elif image_class!=1:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    

    return train_images_path, train_images_label, val_images_path, val_images_label,test_images_path,test_images_label


# 列表删除多个元素的方法：
def del_list_elements(in_list, del_list):
    # list和dict的相互转化的方法,先将list变为dict,再将删除数据的新dict变为新的list并返回
    list_keys = [i for i in range(len(in_list))]
    temp_dict = dict(zip(list_keys, in_list))
    # 存储要删掉的字典的key值
    temp_list = []
    for item in del_list:
        [temp_list.append(k) for k, v in temp_dict.items() if v == item]
    # 根据key值删除dict中的元素
    [temp_dict.pop(i) for i in temp_list]
    # 将dict中的value转为list
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
#     root = '../Data/FAST_DATA'
#     split_fast_data(root)
#     root = "../Data/MedlatTrainingData_png_48x48"
#     read_split_data(root)
    root = '../Data/BNUDataset_48x48'
    split_bnudataset(root)