import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.num_of_weights_per_feat = 1
        self.center = nn.Parameter(torch.randn(self.num_of_weights_per_feat, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        
        w = F.normalize(self.center, p=2, dim=1) # norm on each row of the weight matrix
        x = F.normalize(x, p=2, dim=1) # norm on each row of the input feature matrix
    
        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        if labels is not None:
            scores[labels == 0] = self.r_real - scores[labels == 0] #I changed this line from the original code
            scores[labels == 1] = scores[labels == 1] - self.r_fake #I changed this line from the original code

            loss = self.softplus(self.alpha * scores).mean()
        else:
            loss = None

        return loss , output_scores.squeeze(1)
    
    def inference(self, x):
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        scores = x @ w.transpose(0,1)
        #score = self.softplus(self.alpha *scores)
        return scores