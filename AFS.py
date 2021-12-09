import math
import torch
import torch.nn.functional as F
from torch import nn



class arctanish(nn.Module):
    def __init__(self):
        super(arctanish, self).__init__()

    def forward(self, input):
        return input * (2/math.pi * torch.arctan(input) + 1)/2


######################################################\phi 2
class softsignish(nn.Module):
    def __init__(self):
        super(softsignish, self).__init__()

    def forward(self, input):
        return input * ((input /(1.+torch.abs(input)))+1.)/2 


############################################################\phi 3
class loglogish(nn.Module):
    def __init__(self):
        super(loglogish, self).__init__()

    def forward(self, input):
        return input * (1-torch.exp(-torch.exp(input)))

##############################################################\phi 4
class arctanexp(nn.Module):
    def __init__(self):
        super(arctanexp, self).__init__()

    def forward(self, input):
        return input * (2/math.pi * torch.arctan(torch.exp(input)) )
