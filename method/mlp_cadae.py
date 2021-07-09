import torch
import torch.nn as nn
import torch.nn.functional as F

###Multi_Layer_Perceptrons
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        else:
            #Multi-layer model
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):

        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)