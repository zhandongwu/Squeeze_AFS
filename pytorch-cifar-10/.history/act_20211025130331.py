from torch.nn import functional as F
import math
from torch import nn
class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x =  x * ((2/math.pi * torch.arctan(x) + 1)/2)################################################################# f1/Arctanish
        # x = x * ((x/(1+torch.abs(x)) + 1)/2)########################################################################### f2/Softsignish
        # x = x * (1-torch.exp(-torch.exp(x)))########################################################################### f3/Loglogish
        # x = x * (2/math.pi * torch.arctan(torch.exp(x)))############################################################### f4/ArctanExp
        # x = x * torch.exp(torch.tanh(x)-1)############################################################################# f5 /ExpTanh
        # x = F.relu(x)  ################################################################################################ Relu
        # x = x * F.sigmoid(x)  ######################################################################################### Swish
        # x = x * torch.tanh((F.softplus(x)) )########################################################################### Mish

        return x


    class Mish(nn.Module):
        def __init__(self):
            super(Mish, self).__init__()

        def forward(self, x):
            return x * torch.tanh(F.softplus(x))
        
    class Swish(nn.Module):
        def __init__(self):
            super(Swish, self).__init__()

        def forward(self, x):
            return x * F.sigmoid(x)
        
    class Exptanh(nn.Module):
        def __init_(self):
            super(Exptanh, self).__init__()
            
        def forward(self, x):
            return x * torch.exp(torch.tanh(x)-1)
        
    class Arctanexp(nn.Module):
        def __init_(self):
            super(Arctanexp, self).__init__()
            
        def forward(self, x):
            return x * (2/math.pi * torch.arctan(torch.exp(x)))
        
    class Loglogish(nn.Module):
        def __init_(self):
            super(Loglogish, self).__init__()

        def forward(self, x):
            return x * (1-torch.exp(-torch.exp(x)))
        
    class Arctanish(nn.Module):
        def __init_(self):
            super(Arctanish, self).__init__()

        def forward(self, x):
            return x * ((2/math.pi * torch.arctan(x) + 1)/2)

    class Softsignish(nn.Module):
        def __init_(self):
            super(Softsignish, self).__init__()

        def forward(self, x):
            return x * ((x/(1+torch.abs(x)) + 1)/2)
        
    class Tanhexp(nn.Module):
        def __init_(self):
            super(Tanhexp, self).__init__()

        def forward(self, x):
            return x * (torch.tanh(torch.exp(x)))