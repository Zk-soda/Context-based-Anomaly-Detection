#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:36:01 2020

@author: suzukilab
"""
from __future__ import print_function
import numpy as np
import itertools
import glob
import networkx as nx

from tqdm import tqdm
from collections import defaultdict
import pickle

def def_overlap(box1, box2):
    '''
    compute if two regions are overlapped
    boxes coordinates [a,b,c,d],  (x1,y1) = (a,b), (x2, y2) = (a+c, b+d)
    '''   
    a1,b1,c1,d1 = box1[0], box1[1], box1[2], box1[3]
    a2,b2,c2,d2 = box2[0], box2[1], box2[2], box2[3]
    A = [a1,b1,a1+c1,b1+d1]
    B = [a2,b2,a2+c2,b2+d2]

    iw = min(A[2], B[2]) - max(A[0], B[0])
    if iw > 0:
        ih = min(A[3], B[3]) - max(A[1], B[1])  
        if ih > 0:
            return 1.
        else:
            return 0.
    else:
        return 0.
        
def construct_graph_overlap(im_idx, im_box, im_regionfea, im_capfea, im_captions, im_label):
    '''
    each image constructs one graph, nodes = regions, overlapped = edges
    '''
    ###initalize all fea to 999 to check them at last.
    mx_regionfea = np.ones([im_regionfea.shape[0], im_regionfea.shape[1]])*999
    mx_captionfea = np.ones([im_capfea.shape[0], im_capfea.shape[1]])*999
    mx_captions = np.ones([im_captions.shape[0], im_captions.shape[1]]).astype(im_captions.dtype)
    label = np.ones(im_label.shape[0])*999
    graph = nx.Graph()
    edge_list = []
    total_node = list(range(len(im_idx)))
    adj_lists = defaultdict(set)
    for node_id in total_node:
        graph.add_node(node_id)      
        mx_regionfea[node_id]  = im_regionfea[node_id]
        mx_captionfea[node_id] = im_capfea[node_id]
        mx_captions[node_id]   = im_captions[node_id]
        label[node_id]         = im_label[node_id]
    ###compute if each two regions are overlapped for edge list
    node_pair = list(itertools.combinations(total_node, 2))
    for node_x, node_y in node_pair:
        box_x, box_y = im_box[node_x], im_box[node_y]
        overlap = def_overlap(box_x, box_y)
        if overlap == 1.:
            edge_list.append((node_x,node_y))
            adj_lists[node_x].add(node_y)
            adj_lists[node_y].add(node_x)
    ## add edges from edge list        
    graph.add_edges_from(edge_list)
    ### check if all the features are fullfilled from loaded files
    check1 = [ex for row in mx_regionfea for ex in row if ex == 999]
    check2 = [ex for row in mx_captionfea for ex in row if ex == 999]
    check3 = [ex for row in mx_captions for ex in row if ex == '1.0']
    check4 = [ex for ex in label if ex == 999]
    assert check1 == [] and check2 == [] and check3 == [] and check4 == []
    
    return graph, mx_regionfea, mx_captionfea, mx_captions, adj_lists, label

'''load npz file (kang and hatae data) and construct the graph data 
then save each graph data to the pickle file for each image, where nodes = regions, edges = overlapped,
node attributes = region_fea (imfea/capfea/im&cap_fea).
'''
###kang_data
kang = 1
if kang == 1:
    raw_dir = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE/data/raw_all_labdata/'
    for datatype in ['train','test']:
        kang_imdir = 'selected_'+datatype+'_img/'
        load_data = np.load(raw_dir+'kangdata_npz/'+datatype+'_imfea_capfea_kangdata.npz')
        img_id = [jpg.split('/')[-1] for jpg in glob.glob(raw_dir+kang_imdir+'*.jpg')]

        for im in tqdm(img_id):
            im_idx = list(np.where(load_data['img_names']==im)[0])
            im_regionfea = load_data['region_feature'][im_idx]
            im_caption = load_data['captions'][im_idx].reshape(-1,1)
            im_capfea = load_data['cap_feature'][im_idx]
            im_box = load_data['boxes'][im_idx]
            im_label = load_data['labels'][im_idx]
            ###construct graph data
            im_graph, mx_regionfea, mx_captionfea, mx_captions, adj_list, label \
            = construct_graph_overlap(im_idx, im_box, im_regionfea, im_capfea,
                                      im_caption, im_label)
            ###save each graph with features and labels to each pickle file
            processed_data = {'adj_mx': nx.adjacency_matrix(im_graph),
                              'region_fea': mx_regionfea,
                              'caption_fea':mx_captionfea,
                              'captions':mx_captions,
                              'adj_list':adj_list,
                              'label': label}
            with open(raw_dir+'kangdata_pkl/'+'unnormalize_'+datatype+'/'+\
                      im.split('.')[0]+'.pkl', 'wb') as file:
                pickle.dump(processed_data, file)  
hatae = 0
if hatae == 1:
    hatae_npz = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD\
/GIN_AE/data/toy_hatae_data/hatae_final_im_cap.npz'
    load_data = np.load(hatae_npz)
    
    hatae_data_dir = '/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE/data\
/toy_hatae_data/hatae_totaldata/'
    
    train_imgid = [jpg.split('/')[-1] for jpg in glob.glob(hatae_data_dir+'tr*.jpg')]
    test_imgid  = [jpg.split('/')[-1] for jpg in glob.glob(hatae_data_dir+'te*.jpg')]
    
    img_id = [name for name in set(load_data['img_names'])]
    
    imgset = {'train': train_imgid, 'test': test_imgid}
    for datatype in ['train', 'test']: 
        img_set = imgset[datatype]
        for im in tqdm(img_set):
            im_idx = list(np.where(load_data['img_names']==im)[0])
            im_regionfea = load_data['region_feature'][im_idx]
            im_caption = load_data['captions'][im_idx].reshape(-1,1)
            im_capfea = load_data['cap_feature'][im_idx]
            im_box = load_data['boxes'][im_idx]
            im_label = load_data['labels'][im_idx]
            ###construct graph data
            im_graph, mx_regionfea, mx_captionfea, mx_captions, adj_list, label \
            = construct_graph_overlap(im_idx, im_box, im_regionfea, im_capfea,
                                      im_caption, im_label)
            ###save each graph with features and labels to each pickle file
            processed_data = {'adj_mx': nx.adjacency_matrix(im_graph),
                              'region_fea': mx_regionfea,
                              'caption_fea':mx_captionfea,
                              'captions':mx_captions,
                              'adj_list':adj_list,
                              'label': label}
            with open('/home/suzukilab/research/AD_graph/GraphSage and GIN for AD/GIN_AE/data\
/toy_hatae_data/'+'hatae_'+datatype+'_pkl/'+im.split('.')[0]+'_jpg.pkl', 'wb') as file:
                pickle.dump(processed_data, file)
      