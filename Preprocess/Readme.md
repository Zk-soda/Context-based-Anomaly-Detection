# Preprocess 

After getting jsonfile of captions and bounding boxes from Densecap

## extract_feature_making_npz.py
Extract the visual feature and caption feature by pre-trained ResNet and BERT  
  And save them to npz file


## construct_graph_savepickle.py
Load npz file to get coordinates from bounding boxes of regions  
Compute their spatially overlapped relations and construct graphs  
Save each graph with features to picklefile for each image

## data_normalize.py
Normalize all the data for CADAE



