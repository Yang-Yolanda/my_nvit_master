import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SemGraphConv(nn.Module):
    """
    Semantic Graph Convolution Layer (Inspired by GraphCMR / SemGCN).
    Performs deterministic state transition using a fixed adjacency matrix.
    """
    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Weight Matrix W (Learnable state transition logic)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 2. Adjacency Matrix A (Fixed Knowledge Prior)
        # Registered as buffer so it's saved with state_dict but not updated by optimizer
        self.register_buffer('adj', adj)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        """
        input: [Batch, Nodes, In_Features]
        """
        # H = A * X * W
        
        # Step 1: Feature Transformation (X * W)
        support = torch.matmul(input, self.W) # [B, N, Out]
        
        # Step 2: Structural Aggregation (A * Support)
        # We assume adj is [N, N]. 
        output = torch.matmul(self.adj, support) # [B, N, Out]
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class KinematicGCNBlock(nn.Module):
    """
    Residual Block containing Graph Convolutions.
    Can be used as a drop-in replacement for Transformer Blocks.
    """
    def __init__(self, in_features, hidden_features, adj, dropout=0.1):
        super(KinematicGCNBlock, self).__init__()
        
        # Expansion / Contraction logic can be added here
        # For now, we keep dimension constant for simple residual add
        
        self.gcn1 = SemGraphConv(in_features, hidden_features, adj)
        self.bn1 = nn.BatchNorm1d(adj.shape[0]) # BN over nodes
        self.relu = nn.ReLU(inplace=True)
        
        self.gcn2 = SemGraphConv(hidden_features, in_features, adj) # Project back
        self.bn2 = nn.BatchNorm1d(adj.shape[0])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [Batch, Nodes, Features]
        """
        residual = x
        
        out = self.gcn1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dropout(out)
        
        out = self.gcn2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out
