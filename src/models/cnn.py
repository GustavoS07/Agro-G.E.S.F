import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self,channels,reduction=16):
        super(SEBlock,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential (
            nn.Linear(channels,channels//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels,bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)
    
    
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        
        self.features = nn.Sequential(
            
            #Primeira camada - Convolucional padrão
            nn.Conv2d(3,32,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            #Segunda camada - Depthwise Separable
            nn.Conv2d(32,32,kernel_size=3,padding=1,groups=32,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            #Terceira camda - Depthwise Separable
            nn.Conv2d(64,64,kernel_size=3,padding=1,groups=64,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128, kernel_size=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            #Quarta camada - Depthwise Separable e SE Attention
            nn.Conv2d(128,128,kernel_size=3,padding=1, groups=128,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256,16),
            nn.MaxPool2d(2,2),

            #Quinta camada - Depthwise e SE
            nn.Conv2d(256,256,kernel_size=3,padding=1,groups=256,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512,32),
            nn.MaxPool2d(2,2),

            #Sexta camada 
            nn.Conv2d(512,512,kernel_size=3,padding=1,groups=512,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,768,kernel_size=1,bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            SEBlock(768,48),
            
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Dropout2d(0.3),
        )
                
        self.classifier = nn.Sequential (
            nn.Flatten(),
            nn.Linear(768*4*4,512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )
        self._initialize_weights()         
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x