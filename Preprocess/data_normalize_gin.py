#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:04:30 2020

@author: suzukilab
"""

'''
normalize all data and save them to each pickle file.
'''

from __future__ import print_function
import numpy as np

import glob
from tqdm import tqdm
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
### two normalization ways: standard and minmax
def preprocess(data):
     standard_process = StandardScaler()
     standard_processed_data = standard_process.fit_transform(data)
     
     minmax_process = MinMaxScaler()
     minmax_processed_data = minmax_process.fit_transform(data)
     return standard_processed_data, minmax_processed_data
 
###kang_data
kang = 1   
if kang == 1:
    main_dir = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE/data\
/raw_all_labdata/kangdata_pkl/'
    unor_tr_list = glob.glob(main_dir+'unnormalize_train/*.pkl')
    unor_te_list = glob.glob(main_dir+'unnormalize_test/*.pkl') 
        
    total_list = unor_tr_list + unor_te_list
    
    cap_feature = []
    img_feature = []
    
    for pkl in total_list:
        with open(pkl,'rb') as file:
               data = pickle.load(file) 
        cap_feature.append(data['caption_fea'])
        img_feature.append(data['region_fea'])
    cap_fea_mtx = np.stack(cap_feature).reshape(-1,768)
    im_fea_mtx = np.stack(img_feature).reshape(-1,2048)
    
    stand_capfea_mtx, minmax_capfea_mtx = preprocess(cap_fea_mtx)
    stand_imfea_mtx, minmax_imfea_mtx = preprocess(im_fea_mtx)
    
    stand_capimfea_mtx = np.hstack((stand_capfea_mtx,stand_imfea_mtx))
    minmax_capimfea_mtx = np.hstack((minmax_capfea_mtx,minmax_imfea_mtx))   
    
    save_dir = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD\
/GIN_AE/data/raw_all_labdata/kangdata_pkl/'
    
    for idx, pkl_name in tqdm(enumerate(total_list)):
        
        stand_cap_fea = stand_capfea_mtx[idx*10:(idx+1)*10]
        minmax_cap_fea = minmax_capfea_mtx[idx*10:(idx+1)*10]
        
        stand_im_fea = stand_imfea_mtx[idx*10:(idx+1)*10]
        minmax_im_fea = minmax_imfea_mtx[idx*10:(idx+1)*10]
        
        stand_imcap_fea = stand_capimfea_mtx[idx*10:(idx+1)*10]
        minmax_imcap_fea = minmax_capimfea_mtx[idx*10:(idx+1)*10]
        with open(pkl_name,'rb') as file:
           save_data = pickle.load(file) 
        save_data['cap_stand_fea'] = stand_cap_fea
        save_data['cap_minmax_fea'] = minmax_cap_fea
        
        save_data['im_stand_fea'] = stand_im_fea
        save_data['im_minmax_fea'] = minmax_im_fea
        
        save_data['imcap_stand_fea'] = stand_imcap_fea
        save_data['imcap_minmax_fea'] = minmax_imcap_fea
        with open(save_dir+pkl_name.split('/')[-2].split('_')[1]+'/'+pkl_name.split('/')[-1], 'wb') as save_file:
            pickle.dump(save_data, save_file)
            
hatae_toy_data = 0            
if hatae_toy_data == 1:
    unor_tr_list = glob.glob('/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE\
/data/toy_hatae_data/unormalize_train_pkl/*.pkl')
    unor_te_list = glob.glob('/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE\
/data/toy_hatae_data/unormalize_test_pkl/*.pkl') 
    total_list = unor_tr_list + unor_te_list
    
    cap_feature = []
    img_feature = []
    
    for pkl in total_list:
        with open(pkl,'rb') as file:
               data = pickle.load(file) 
        cap_feature.append(data['caption_fea'])
        img_feature.append(data['region_fea'])
    cap_fea_mtx = np.stack(cap_feature).reshape(-1,768)
    im_fea_mtx = np.stack(img_feature).reshape(-1,2048)
    
    stand_capfea_mtx, minmax_capfea_mtx = preprocess(cap_fea_mtx)
    stand_imfea_mtx, minmax_imfea_mtx = preprocess(im_fea_mtx)
    
    stand_capimfea_mtx = np.hstack((stand_capfea_mtx,stand_imfea_mtx))
    minmax_capimfea_mtx = np.hstack((minmax_capfea_mtx,minmax_imfea_mtx))   
    
    save_dir = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD\
/GIN_AE/data/toy_hatae_data/'
    
    for idx, pkl_name in tqdm(enumerate(total_list)):
        
        stand_cap_fea = stand_capfea_mtx[idx*10:(idx+1)*10]
        minmax_cap_fea = minmax_capfea_mtx[idx*10:(idx+1)*10]
        
        stand_im_fea = stand_imfea_mtx[idx*10:(idx+1)*10]
        minmax_im_fea = minmax_imfea_mtx[idx*10:(idx+1)*10]
        
        stand_imcap_fea = stand_capimfea_mtx[idx*10:(idx+1)*10]
        minmax_imcap_fea = minmax_capimfea_mtx[idx*10:(idx+1)*10]
        with open(pkl_name,'rb') as file:
           save_data = pickle.load(file) 
        save_data['cap_stand_fea'] = stand_cap_fea
        save_data['cap_minmax_fea'] = minmax_cap_fea
        
        save_data['im_stand_fea'] = stand_im_fea
        save_data['im_minmax_fea'] = minmax_im_fea
        
        save_data['imcap_stand_fea'] = stand_imcap_fea
        save_data['imcap_minmax_fea'] = minmax_imcap_fea
        with open(save_dir+pkl_name.split('/')[-2].split('_')[1]+'/'+pkl_name.split('/')[-1], 'wb') as save_file:
            pickle.dump(save_data, save_file)            
toy_data = 0
if toy_data == 1:
    tr_list = glob.glob('./data/toy_labdata/batch_graph_region&caption/train/*.pkl')
    te_list = glob.glob('./data/toy_labdata/batch_graph_region&caption/test/*.pkl') 
    
    total_list = tr_list+te_list
    
    feature = []
    for pkl in total_list:
        with open(pkl,'rb') as file:
               data = pickle.load(file) 
        feature.append(data['caption_fea'])
        
    fea_mtx = np.stack(feature).reshape(-1,768)
    stand_fea_mtx, minmax_fea_mtx = preprocess(fea_mtx)
    
    save_dir = './data/toy_labdata/batch_graph_region&caption_normalize'
    
    for idx, pkl_name in enumerate(total_list):
        
        stand_fea = stand_fea_mtx[idx*10:(idx+1)*10]
        minmax_fea = minmax_fea_mtx[idx*10:(idx+1)*10]
        with open(pkl_name,'rb') as file:
           save_data = pickle.load(file) 
        save_data['cap_stand_fea'] = stand_fea
        save_data['cap_minmax_fea'] = minmax_fea
        with open(save_dir+pkl_name.split('batch_graph_region&caption')[-1], 'wb') as save_file:
            pickle.dump(save_data, save_file)
            