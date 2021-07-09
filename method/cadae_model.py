import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp_cadae import MLP
from innerproductdecoder import InnerProductDecoder


class Base_CADAE(nn.Module):
    def __init__(self, num_layers, dec_num_layers, num_mlp_layers, input_dim, hidden_dim, embedd_dim, str_recon_dim, learn_eps, neighbor_pooling_type, dropout, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            dec_num_layers: umber of layers in the decoder (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            embedd_dim: dimensionality of output embeedings from encoder
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors
        '''

        super(Base_CADAE, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.dec_num_layers = dec_num_layers

        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.dropout = dropout
        
        ###encoder
        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, embedd_dim, embedd_dim))
                self.batch_norms.append(nn.BatchNorm1d(embedd_dim))

            
        ###attribute decoder 
        self.dec_mlps = torch.nn.ModuleList()
        self.dec_batch_norms = torch.nn.ModuleList()
        for layer in range(self.dec_num_layers-1):
            if layer == 0:
                self.dec_mlps.append(MLP(num_mlp_layers, embedd_dim, embedd_dim, hidden_dim))
                self.dec_batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.dec_mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, input_dim))
       
            
        ###structure decoder
        self.dec_str_mlp = MLP(num_mlp_layers, embedd_dim, hidden_dim, str_recon_dim)
        ## innerproduct for reconstructing adjacent matrix
        self.innerproduct = InnerProductDecoder(self.dropout)


    def neighbors_sumavepool(self, adj_ori):
        
        if self.learn_eps == True:
            return adj_ori
        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
        elif not self.learn_eps:
            if self.training:
                adj_selfloop = torch.cuda.FloatTensor(adj_ori+torch.eye(adj_ori.shape[0]).cuda())
            else:
                adj_selfloop = torch.FloatTensor(adj_ori+torch.eye(adj_ori.shape[0]))
            return adj_selfloop
###encoder next layers

    def next_layer_eps(self, h, layer, Adj_block = None):
        ###if eps=True, pooling neighboring nodes and center nodes separately by eps reweighting. 

        pooled = torch.mm(Adj_block, h)

        ### aggregate representations of node and its neighbors by the parameter and sum aggregation
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###if eps = False, pooling neighboring nodes and center nodes altogether  
  
        pooled = torch.mm(Adj_block, h)
    
        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity, last layer without batchnorm and relu
        h = F.relu(h)
        return h
    
###decoder next layers
    def dec_next_layer_eps(self, h, layer, Adj_block = None, last = False):
        ###same with encoder layers

        pooled = torch.mm(Adj_block, h)

        pooled = pooled + (1 + self.eps[layer])*h 
        pooled_rep = self.dec_mlps[layer](pooled)

        #non-linearity
        if last == False:
            h = self.batch_norms[layer](pooled_rep)
            h = F.relu(h)
        else:
            h = pooled_rep
        return h

    def dec_next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None, last = False):
        ###same with encoder layers
            
        pooled = torch.mm(Adj_block, h)

        #representation of neighboring and center nodes 
        pooled_rep = self.dec_mlps[layer](pooled)
        #non-linearity, last layer without batchnorm and relu
        if last == False:
            h = self.batch_norms[layer](pooled_rep)
            h = F.leaky_relu(h)
        else:
            h = pooled_rep
        return h
    
    def forward(self, batch_graph, adj_ori):

        Adj_block = self.neighbors_sumavepool(adj_ori)

        #matrix of hidden representation at each layer (including input)
        
        h = batch_graph

        for layer in range(self.num_layers-1):
            if self.learn_eps ==True:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)
                
        #hidden embeddings
        hidden_embs = h
        
        for layer in range(self.dec_num_layers-1):
            if layer != self.dec_num_layers - 2:
                if self.learn_eps ==True:
                    h_1 = self.dec_next_layer_eps(hidden_embs, layer, Adj_block = Adj_block, last=False)
                elif not self.learn_eps:
                    h_1 = self.dec_next_layer(hidden_embs, layer, Adj_block = Adj_block, last=False)
            elif layer == self.dec_num_layers - 2:
                if self.learn_eps ==True:
                    h = self.dec_next_layer_eps(h_1, layer, Adj_block = Adj_block, last=True)
                elif not self.learn_eps:
                    h = self.dec_next_layer(h_1, layer, Adj_block = Adj_block, last=True)
            else:
                raise ValueError('not valid layer number!')
                
        att_recons = h
        #reconstruct adjacent matrix (structure)
        str_embs   = self.dec_str_mlp(hidden_embs)
        str_recons = self.innerproduct(str_embs)

        return att_recons, str_recons, hidden_embs
   
