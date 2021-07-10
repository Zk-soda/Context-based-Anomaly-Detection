# Preprocess 
## pipeline

Raw Image --> jsonfile of captions and bounding boxes by Densecap --> npzfile of extracting visual and caption features by ResNet and BERT --> picklefile of constructed graphs for images including regions as nodes and node attributes as region features --> data normalization  
**pretrained models** can be found from:  
[Densecap](https://github.com/jcjohnson/densecap)  
[ResNet](https://pytorch.org/vision/stable/models.html)  
[BERT](https://github.com/hanxiao/bert-as-service)
## extract_feature_making_npz.py
Extract the visual features of regions and caption features by pre-trained ResNet and BERT  
And save them to npz file

## construct_graph_savepickle.py
Load npz file to get coordinates from bounding boxes of regions  
Compute their spatially overlapped relations and construct graphs  
Save each graph with features to picklefile for each image

## data_normalize.py
Normalize all the data for CADAE



