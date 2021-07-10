#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:36:01 2020

@author: suzukilab
"""
from __future__ import print_function
import numpy as np
import shutil
from time import time
import glob
import json
import os
from PIL import Image

import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

def extract_feature(path, data, model, input_x, device):
    
    ##extract image features from the second last output layer of the model
    
    extract_model = nn.Sequential(*list(model.children())[:-1])
    '''
    for cuda
    '''
    extract_model = extract_model.to(device)
    input_x = input_x.unsqueeze(0).to(device)

    
    feature = extract_model(input_x)
    output_feature = feature.view(input_x.size(0), -1)
    
    output_feature = output_feature.detach().cpu().numpy()
    return output_feature

def preprocess_img(path, data, model, device):
    ###process the regions and extract features
    t = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
    img_dir = [path + name for name in data['img_names']]
    img_box = [box for box in data['boxes']]
    
    im_feature = np.empty(shape= [0,2048])
    for img, cor in tqdm(zip(img_dir, img_box)):
        cor = np.maximum(cor, 0) #delete negative coordinates from boxes
        #crop reigon
        region = Image.open(img).crop((cor[0], cor[1], cor[0]+cor[2], cor[1]+cor[3])) 
        process_img = t(region)
        
        feature = extract_feature(path, data, model, process_img, device)
        im_feature = np.append(im_feature, feature, axis = 0)
    return im_feature

def loadjson_kang_data(path):
    #load name, boundingbox, caption ,label from jsonfile
    data_name = dict()
    boxes = dict()
    captions = dict()
    total_lab = dict()
    assert os.listdir(path)[4] == 'selected_test_img' and os.listdir(path)[5] == 'selected_train_img'
    for sub_path in [os.listdir(path)[4].split('_')[1],os.listdir(path)[5].split('_')[1]]:
        jpg_list = glob.glob(os.path.join(path, 'selected_'+sub_path+'_img', '*.jpg'))
        name = [jpg_name.split('/')[-1] for jpg_name in jpg_list]        
        num_sample = 10
        total_lab[sub_path] = np.empty(shape = [0])
        data_name[sub_path] = []
        boxes[sub_path] = []
        captions[sub_path] = []
        json_dir = 'selected_'+sub_path+'_json(rename)'
        for j in name:
            jsonfile = json.load(open(os.path.join(path, json_dir, j.replace('jpg', 'json'))))
            results = jsonfile['results'][0]
            for name_copy in range(num_sample):
                data_name[sub_path].append(results['new_img_name'])
            boxes[sub_path].extend(results['boxes'][:num_sample])
            captions[sub_path].extend(results['captions'][:num_sample])
            
            if 'labels' in results.keys():
                assert len(results['labels']) == 10
                total_lab[sub_path] = np.append(total_lab[sub_path], results['labels'])
            else:
                normal_lab = np.zeros(num_sample,dtype = int)
                total_lab[sub_path] = np.append(total_lab[sub_path], normal_lab)
    return data_name, boxes, captions, total_lab

def load_rename_json_hatae_data(path, img_dir, save_folder, data_type='te', save_img_json=True):
    #move jsonfile to train and text, load name, boundingbox, caption ,label from jsonfile
    jsonfile = json.load(open(path))
    data_name = np.array(list(jsonfile['img_name'].values()))
    boxes = np.array(list(jsonfile['boxes'].values()))
    captions = np.array(list(jsonfile['captions'].values()))
    labels = np.array(list(jsonfile['labels'].values()))
    
    ###delete not proper image
    if data_type == 'te':
        for delete_jpg in ['1.abnormal/155.jpg','1.abnormal/158.jpg','1.abnormal/187.jpg']:
            delete_idx = np.where(data_name == delete_jpg)[0]
            
            data_name = np.delete(data_name, delete_idx)
            boxes = np.delete(boxes, delete_idx, axis=0)
            captions = np.delete(captions, delete_idx)
            labels = np.delete(labels, delete_idx)
    
    ###rename img_name and save image and json file to abnormal and normal folder.
    if save_img_json == True:
        total_data_name = set(data_name)
        for num, name in enumerate(total_data_name):
            if name.split('.')[0]=='1':
                im_idx = np.where(data_name==name)[0]
                json_data={'results':list()}
                json_data['results'].append({'img_name':data_type+'_hatae_ab_'+str(num)+'.jpg'})
                json_data['results'].append({'captions':list(captions[im_idx])})
                json_data['results'].append({'boxes':list(list(b) for b in boxes[im_idx])})
                json_data['results'].append({'labels':list(int(l) for l in labels[im_idx])})
                save_json = save_folder+data_type+'_hatae_ab_'+str(num)+'.json'
                with open(save_json,'w') as file:
                    json.dump(json_data, file)
                
                data_name[data_name==name] = data_type+'_hatae_ab_'+str(num)+'.jpg'
                assert glob.glob(img_dir+name) != []
                shutil.copy(img_dir+name, save_folder+data_type+'_hatae_ab_'+str(num)+'.jpg')
                
            else:
                assert name.split('.')[0]=='0'
                im_idx = np.where(data_name==name)[0]
                json_data={'results':list()}
                json_data['results'].append({'img_name':data_type+'_hatae_n_'+str(num)+'.jpg'})
                json_data['results'].append({'captions':list(captions[im_idx])})
                json_data['results'].append({'boxes':list(list(b) for b in boxes[im_idx])})
                json_data['results'].append({'labels':list(int(l) for l in labels[im_idx])})
                save_json = save_folder+data_type+'_hatae_ab_'+str(num)+'.json'
                with open(save_json,'w') as file:
                    json.dump(json_data, file)
                    
                data_name[data_name==name] = data_type+'_hatae_n_'+str(num)+'.jpg'
                assert glob.glob(img_dir+name) != []
                shutil.copy(img_dir+name, save_folder+data_type+'_hatae_n_'+str(num)+'.jpg')
    
    return data_name, boxes, captions, labels

def add_capfeature(path, datatype, ori_data = 'total', bert = 1, data='kang'):
    if data == 'toy':
        load_data = np.load(ori_data+'_labdata.npz', allow_pickle=True)
        if bert == 1:
            from bert_serving.client import BertClient
        encoder = BertClient()
        cap_feature = np.empty(shape=[0,768])
        for i in tqdm(load_data['captions']):
            cap_feature = np.append(cap_feature, encoder.encode([i]), axis = 0)
        fea_cap_region = np.concatenate((load_data['region_feature'], cap_feature), 
                                        axis = 1)
        datatype = 'total_cr'
        
        np.savez(datatype+'_labdata.npz', img_names = load_data['img_names'], 
                 captions = load_data['captions'], boxes = load_data['boxes'], 
                 labels = load_data['labels'], region_feature = load_data['region_feature'],
                 cap_feature = cap_feature, cap_region_fea = fea_cap_region)
        
    if data == 'hatae':
        load_data = np.load(path)
        if bert == 1:
            from bert_serving.client import BertClient
        encoder = BertClient()
        cap_feature = np.empty(shape=[0,768])
        for i in tqdm(load_data['captions']):
            cap_feature = np.append(cap_feature, encoder.encode([i]), axis = 0)
        return cap_feature
    
    if data == 'kang':
        load_data = np.load(path+'kangdata_npz/'+datatype+'_imfea_kangdata.npz')
        if bert == 1:
            from bert_serving.client import BertClient
        encoder = BertClient()
        cap_feature = np.empty(shape=[0,768])
        for i in tqdm(load_data['captions']):
            cap_feature = np.append(cap_feature, encoder.encode([i]), axis = 0)
        fea_cap_region = np.concatenate((load_data['region_feature'], cap_feature), 
                                        axis = 1)
        np.savez(path+'kangdata_npz/'+datatype+'_imfea_capfea_kangdata.npz', img_names = load_data['img_names'], 
                 captions = load_data['captions'], boxes = load_data['boxes'], 
                 labels = load_data['labels'], region_feature = load_data['region_feature'],
                 cap_feature = cap_feature, cap_region_fea = fea_cap_region)     
        
np.set_printoptions(suppress=True)


kang_data = 1
hatae_data = 0
if kang_data == 1:
    '''First save raw npz file from loading json of kangdata, then use pretrained resnet and bert to 
    extract imfea and caption features to the total npz file.
       
    '''
    ###set to 1 to run each step
    save_raw = 0
    add_imfea = 0
    add_cap = 0
    kang_data_path = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE/data/raw_all_labdata/'
    data_name, boxes, captions, total_lab = loadjson_kang_data(kang_data_path)
    
    if save_raw ==1: 
        for datatype in data_name.keys():
            np.savez(kang_data_path+'kangdata_npz/'+datatype+'_kangdata.npz', img_names = data_name[datatype], 
                 captions = captions[datatype], boxes = boxes[datatype], 
                 labels = total_lab[datatype])
    
    if add_imfea == 1:
        for datatype in data_name.keys():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            resnet101 = models.resnet101(pretrained = True)
            
            load_data = np.load(kang_data_path+'kangdata_npz/'+datatype+'_kangdata.npz', allow_pickle=True)
            load_path = os.path.join(kang_data_path, 'selected_'+datatype+'_img')+'/'
            time1=time()
            region_feature = preprocess_img(load_path, load_data, resnet101, device)
            time2 = time()      
            print('time for all reigons', datatype+':',time2-time1)
            np.savez(kang_data_path+'kangdata_npz/'+datatype+'_imfea_kangdata.npz', img_names = data_name[datatype], 
                 captions = captions[datatype], boxes = boxes[datatype], 
                 labels = total_lab[datatype], region_feature = region_feature)
            
    if add_cap == 1:
        for datatype in data_name.keys():
            add_capfeature(kang_data_path,datatype,bert=1, data='kang')

if hatae_data == 1:
    ##same setting with kang_data
    save_hatae_npz = 0
    hatae_add_imfea = 0 
    hatae_add_capfea = 0
    if save_hatae_npz == 1:
        
        te_data_path = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/test/test.json'
        te_img_dir = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/test/'
        te_save_folder = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/test\
    /processed_testforGIN/'
    
        tr_data_path = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/train/train.json'
        tr_img_dir = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/train/'
        tr_save_folder = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/train\
    /processed_trainforGIN/'
        te_data_name, te_boxes, te_captions, te_labels = load_rename_json_hatae_data(te_data_path, 
                                                                                     te_img_dir, 
                                                                                     te_save_folder,
                                                                                     data_type = 'te',
                                                                                     save_img_json=True)
        
        tr_data_name, tr_boxes, tr_captions, tr_labels = load_rename_json_hatae_data(tr_data_path, 
                                                                                     tr_img_dir, 
                                                                                     tr_save_folder,
                                                                                     data_type = 'tr',
                                                                                     save_img_json=True)
        total_data_name = np.vstack((tr_data_name.reshape(-1,1), te_data_name.reshape(-1,1))).reshape(-1)
        total_captions = np.vstack((tr_captions.reshape(-1,1), te_captions.reshape(-1,1))).reshape(-1)
        total_boxes = np.vstack((tr_boxes, te_boxes)).reshape(-1,tr_boxes.shape[1])
        total_labels = np.vstack((tr_labels.reshape(-1,1), te_labels.reshape(-1,1))).reshape(-1)
        np.savez('/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/hatae_totaldata.npz', img_names = total_data_name, 
                 captions = total_captions, boxes = total_boxes, 
                 labels = total_labels)
    if hatae_add_imfea == 1 and hatae_add_capfea == 1:
        load_data_name = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master\
    /data_1/hatae_totaldata.npz'
        load_img_dir = '/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1\
    /hatae_totaldata/'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resnet101 = models.resnet101(pretrained = True)
        
        load_data = np.load(load_data_name, allow_pickle=True)
        #load_path = os.path.join(data_path, datatype)+'/'
        time1=time()
        region_features = preprocess_img(load_img_dir, load_data, resnet101, device)
        time2 = time()
        print('time for all reigons:', time2-time1)
        
        cap_features = add_capfeature(load_data_name, ori_data = 'total', bert = 1, data='hatae')
        np.savez('/home/suzukilab/research/AD_graph/GCN_AnomalyDetection-master/data_1/hatae_add_imfea.npz', 
                 img_names = load_data['img_names'], 
                 captions = load_data['captions'], boxes = load_data['boxes'], 
                 labels = load_data['labels'], region_feature = region_features, 
                 cap_feature = cap_features)
    
