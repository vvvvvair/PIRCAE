import os

import torch
from torchvision import transforms
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from my_dataset import MyDataSet
from utils_split import split_Medlat，split_PMPS
from AE import R_AE1, BasicBlock
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from PIL import Image
import random





def data_loader():
    
    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(root)
    train_images_path_ints = [path.replace("bands","ints") for path in  train_images_path]
    val_images_path_ints = [path.replace("bands","ints") for path in  val_images_path]
    test_images_path_ints = [path.replace("bands","ints") for path in  test_images_path]

    
    test_images_path_numpy = np.array(test_images_path)
    test_images_label_numpy = np.array(test_images_label)
    test_images_path_pulsar = list(test_images_path_numpy[test_images_label_numpy == 1])
    test_images_label_pulsar = list(test_images_label_numpy[test_images_label_numpy == 1])

    test_images_path_pulsar_ints =  [path.replace("bands","ints") for path in  test_images_path_pulsar]

    data_transform = {
        "data": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),  
        ]),
        "test":transforms.Compose([
            transforms.ToTensor(),
            
            
        ])}
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["data"])
    train_intsdata_set = MyDataSet(images_path = train_images_path_ints,
                                   images_class = train_images_label,
                                   transform = data_transform['data']
    )
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["data"])
    val_intsdata_set = MyDataSet(images_path=val_images_path_ints,
                                images_class = val_images_label,
                                transform = data_transform['data'])
    test_data_pulsar = MyDataSet(
        images_path=test_images_path_pulsar,
        images_class=test_images_label_pulsar,
        transform=data_transform["test"])
    test_data_set = MyDataSet(
        images_path = test_images_path,
        images_class = test_images_label,
        transform = data_transform["test"]
    )
    test_intsdata_set = MyDataSet(images_path =test_images_path_ints,
                                 images_class = test_images_label,
                                 transform = data_transform['test'])

    batch_size = 256
    train_data_loader =  torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               collate_fn=train_data_set.collate_fn,
                                               )
    train_ints_loader = torch.utils.data.DataLoader(train_intsdata_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               collate_fn=train_intsdata_set.collate_fn,
                                               )
    val_data_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             collate_fn=val_data_set.collate_fn,
                                            drop_last = True
                                             )
    val_ints_loader = torch.utils.data.DataLoader(val_intsdata_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             collate_fn=val_intsdata_set.collate_fn,
                                             )
    test_data_pulsar_loader = torch.utils.data.DataLoader(test_data_pulsar,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             collate_fn=test_data_pulsar.collate_fn,
                                            drop_last = True  
                                            )
  
    
    test_data_loader = torch.utils.data.DataLoader(test_data_set,
                                                   batch_size = 1,
                                                   shuffle = False,
                                                   num_workers = 8,
                                                   collate_fn=test_data_set.collate_fn,
                                                   
    )
    test_intsdata_loader = torch.utils.data.DataLoader(test_intsdata_set,
                                                   batch_size = batch_size,
                                                   shuffle = False,
                                                   num_workers = 8,
                                                   collate_fn=test_intsdata_set.collate_fn,
    )
    
    
    return train_data_loader,val_data_loader,test_data_pulsar_loader,test_data_loader

def train():
    
    model_ae = R_AE1(BasicBlock).to(device)
    
    optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=0.001)
    loss_func_ae = nn.MSELoss(reduction='sum')

    f1_max = 0.95
    for epochs in range(20):
        max_error_list = []
        model_ae.train()
        train_loss = 0
        val_restruct_loss = []
        val_test_loss = []
        for step2,(images2,labels2) in enumerate(train_data_loader):
            encoder,decoder = model_ae(images2.to(device))
            loss = loss_func_ae(decoder,images2.to(device))
            train_loss += loss.data
            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            if (step2 + 1) % 100 == 0:
                print("Epoch:{},train_loss:{}".format(epochs,loss.data/len(images2))) 
            
        model_ae.eval()
        with torch.no_grad():
            for step,(images,labels) in  enumerate(test_data_loader):
                encoder,decoder = model_ae(images.to(device))
                encoder = torch.flatten(encoder,1)
                if step==0:
                    encoder_feature_numpy = encoder
                    labels_numpy = labels
                else:
                    encoder_feature_numpy = torch.cat((encoder_feature_numpy,encoder),dim=0)
                    labels_numpy = torch.cat((labels_numpy,labels),dim = 0)

            for step,(images,labels) in  enumerate(train_data_loader):
                encoder,decoder = model_ae(images.to(device))
                encoder = torch.flatten(encoder,1)
                if step==0:
                    val_encoder_feature_numpy = encoder
                    val_labels_numpy = labels
                else:
                    val_encoder_feature_numpy = torch.cat((encoder_feature_numpy,encoder),dim=0)
                    val_labels_numpy = torch.cat((labels_numpy,labels),dim = 0)
        
        encoder_feature_numpy = encoder_feature_numpy.cpu().numpy()
        labels_numpy = labels_numpy.cpu().numpy().ravel()
        
        val_encoder_feature_numpy = val_encoder_feature_numpy.cpu().numpy()
        val_labels_numpy = val_labels_numpy.numpy().ravel()
        
        
        
        lr = LogisticRegression(max_iter = 10000,random_state=0,class_weight={0:0.25,1:0.75})

        strKFolad = StratifiedKFold(n_splits=10,shuffle=False)
        acc_sum = 0 
        pre_sum = 0
        rec_sum = 0
        f1_sum = 0
        for train_index,test_index in strKFolad.split(encoder_feature_numpy,labels_numpy):
            
            X_train,X_test = encoder_feature_numpy[train_index],encoder_feature_numpy[test_index]
            y_train,y_test = labels_numpy[train_index],labels_numpy[test_index]
            
            X_train = np.vstack([X_train,val_encoder_feature_numpy])
            y_train = np.append(y_train,val_labels_numpy)
            
            
            model_lr = lr.fit(X_train,y_train)
            print("****")
            
            acc = accuracy_score(y_test,model_lr.predict(X_test))
            pre = precision_score(y_test,model_lr.predict(X_test))
            rec = recall_score(y_test,model_lr.predict(X_test))
            f1 = f1_score(y_test,model_lr.predict(X_test))
            acc_sum += acc
            pre_sum += pre
            rec_sum += rec
            f1_sum +=f1
            print(confusion_matrix(y_test,model_lr.predict(X_test)))
            
            print(acc,pre,rec,f1)
        if f1_sum/10>f1_max:
            f1_max = f1_sum/10
            save_path = './{}Medlat_RAE1_best_f1_2022_8_10.pth'
            torch.save(model_ae.state_dict(),save_path.format(epochs))
            with open('./lr_Medlat_RAE1_best_f1_2022_8_10.pickle','wb') as f:
                pickle.dump(lr,f)
        print("****")
        print("****")
        print(acc_sum/10,pre_sum/10,rec_sum/10,f1_sum/10)
            
            
def test():
    
    model_ae = R_AE1(BasicBlock).to(device)
    weight_path = './4Medlat_RAE1_best_f1_2022_8_10.pth'
    model_ae.load_state_dict(torch.load(weight_path))
    model_ae.eval()
    with torch.no_grad():
        for step,(images,labels) in  enumerate(test_data_loader):
            encoder,decoder = model_ae(images.to(device))
            encoder = torch.flatten(encoder,1)
            if step==0:
                encoder_feature_numpy = encoder
                labels_numpy = labels
            else:
                encoder_feature_numpy = torch.cat((encoder_feature_numpy,encoder),dim=0)
                labels_numpy = torch.cat((labels_numpy,labels),dim = 0)
               
        for step,(images,labels) in  enumerate(train_data_loader):
            
            encoder,decoder = model_ae(images.to(device))
            encoder = torch.flatten(encoder,1)
            if step==0:
                val_encoder_feature_numpy = encoder
                val_labels_numpy = labels
            else:
                val_encoder_feature_numpy = torch.cat((encoder_feature_numpy,encoder),dim=0)
                val_labels_numpy = torch.cat((labels_numpy,labels),dim = 0)
        
        encoder_feature_numpy = encoder_feature_numpy.cpu().numpy()
        labels_numpy = labels_numpy.cpu().numpy().ravel()
        
        val_encoder_feature_numpy = val_encoder_feature_numpy.cpu().numpy()
        val_labels_numpy = val_labels_numpy.numpy().ravel()
        
        
                
       
        lr = LogisticRegression(max_iter = 10000,random_state=0,class_weight={0:0.25,1:0.75})

        strKFolad = StratifiedKFold(n_splits=10,shuffle=False)
        acc_sum = 0 
        pre_sum = 0
        rec_sum = 0
        f1_sum = 0
        for train_index,test_index in strKFolad.split(encoder_feature_numpy,labels_numpy):
            
            X_train,X_test = encoder_feature_numpy[train_index],encoder_feature_numpy[test_index]
            y_train,y_test = labels_numpy[train_index],labels_numpy[test_index]
            
            X_train = np.vstack([X_train,val_encoder_feature_numpy])
            y_train = np.append(y_train,val_labels_numpy)
            
            model_lr = lr.fit(X_train,y_train)
            print("****")
            
            acc = accuracy_score(y_test,model_lr.predict(X_test))
            pre = precision_score(y_test,model_lr.predict(X_test))
            rec = recall_score(y_test,model_lr.predict(X_test))
            f1 = f1_score(y_test,model_lr.predict(X_test))
            acc_sum += acc
            pre_sum += pre
            rec_sum += rec
            f1_sum +=f1
            print(confusion_matrix(y_test,model_lr.predict(X_test)))
            
            print(acc,pre,rec,f1)
        print(acc_sum/10,pre_sum/10,rec_sum/10,f1_sum/10)
    
    

if __name__=="__main__":
    root = "../Data/MedlatTrainingData_png_48x48"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_loader, val_data_loader, test_data_pulsar_loader,test_data_loader = data_loader()
    print("using {} device.".format(device))
    train()
    #test()

    
