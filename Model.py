from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, groups=planes,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
class DownH(nn.Module):
    def __init__(self, in_planes, out_planes):
    	super(DownH, self).__init__()
    	self.Conv=nn.Sequential(nn.Conv2d(in_planes,out_planes,3,1,1),
    	nn.MaxPool2d((2,1),(2,1)),
		nn.BatchNorm2d(out_planes),
    	nn.ReLU(True))
    def forward(self,x):
    	out=self.Conv(x)
    	
    	return out

class NetCrnn(nn.Module):
    def __init__(self, num_class=64,batchSize=10,useCuda=True,w=512,h=64,c=3):
        super(NetCrnn, self).__init__()
        self.num_class = num_class
        self.batch=batchSize
        self.useCuda=useCuda
        main=nn.Sequential()
        main.add_module("Conv",nn.Conv2d(3,16,5,2,2))
        w,h=int(w/2),int(h/2)
        multi=(int)(w/h)
        c=16
        
        while(int(h/2)>1 and int(w/2)>32):
        	main.add_module('conv-{0}-{1}'.format(c, c*2),Block(c,c*2,3,2))
        	w,h=int(w/2),int(h/2)
        	c=c*2
        	main.add_module('conv-{0}-{1}'.format(c, c),DownH(c,c))
        	main.add_module('pool-{0}-{1}'.format(c, c),nn.Conv2d(c,c*2,1,1))
        	c=c*2
        	h=int(h/2)
        
        main.add_module('conv-{0}-{1}'.format(c, c*2),Block(c,c*2,4,2))
        c=c*2
        self.main=main
        self.c=c
 
        self.num_layers=5 
        self.Lstm=nn.LSTM(c,c,2,True,True,bidirectional=True)
        self.fc=nn.Linear(self.c*2,self.num_class)
        self.hid=self.init_hidden(self.batch,c*2)


    def init_hidden(self, batch_size,c):
        h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, c))
        if(self.useCuda):
        	return h0.cuda()
        else:
        	return h0.cpu()

    def forward(self, x):
        Feature=self.main(x)
        Feature=Feature.squeeze(2)
        Feature=Feature.transpose(1,2)
        out,hid = self.Lstm(Feature)
        out=out.reshape(-1,self.c*2)
        fc=self.fc(out)
        return fc


#alph="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,"
#c=len(alph)
net=NetCrnn(64,1,False).cuda()
net=net.cuda()
x=torch.randn(1,3,64,512,device='cuda')#[03457]
#label=torch.from_numpy(np.array([[0,3,4,5,7,7,9,3,1]])).cuda()
#input_length=torch.from_numpy(np.array([32])).cuda()
#target_length=torch.from_numpy(np.array([9])).cuda()
y=net(x)
#y=y.transpose(0,1)
#ctc_loss = nn.CTCLoss(blank=12, reduction='mean')
#loss=ctc_loss(y.log_softmax(2),label,input_length,target_length)


