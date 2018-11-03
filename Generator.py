import torch
import torch.nn as nn
from ops import *
from torch.autograd import Variable

def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v =='M':
            layers += [nn.MaxPool2d(2),]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size = 3, stride = 1, padding = 1), relu(inplace = True),]
            in_channels = v
    print(*layers)
    return nn.Sequential(*layers)

def make_deconv_layers(cfg):
    layers = []
    in_channels = 512
    for v in cfg:
        if v =='U':
            layers += [nn.Upsample(scale_factor = 2),]
        else :
            layers += [nn.ConvTranspose2d(in_channels, v, kernel_size = 3, stride = 1, padding = 1),]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
      }

def encoder():
    return make_conv_layers(cfg['E'])
def decoder():
    return make_deconv_layers(cfg['D'])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.mymodules = nn.ModuleList([nn.ConvTranspose2d( 64, 1, kernel_size = 1,stride =1, padding = 0), nn.Sigmoid()])
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.mymodules[0](x)
        x = self.mymodules[1](x)
        return x
 
#This section was just used for checking the working  
# g = Generator()
# x = Variable(torch.rand([17, 3, 192, 256]))
# print('i/P: ', x.size())
# out = g(x)
# print('ó:', out.size())
                                        
    
            
    
