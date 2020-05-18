from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
class ResidualBlock(nn.Module):
	def __init__(self,i_channel,o_channel,stride=1,downsample=None):
		super().__init__()
		self.conv1=nn.Conv2d(i_channel,o_channel,3,stride,1)
		self.bn1=nn.BatchNorm2d(o_channel)
		self.relu1=nn.ReLU(True)

		self.conv2=nn.Conv2d(in_channels=o_channel,out_channels=o_channel,kernel_size=3,stride=1,padding=1,bias=False)        
		self.bn2=nn.BatchNorm2d(o_channel)        
		self.downsample=downsample
	def forward(self,x):        
		residual=x                
			
		out=self.conv1(x)        
		out=self.bn1(out)        
		out=self.relu1(out)        
		out=self.conv2(out)        
		out=self.bn2(out)                
			
		if self.downsample:            
			residual=self.downsample(x)                
		out+=residual        
		out=self.relu1(out)                
		return out


class NetCrnn(nn.Module):
    def __init__(self, num_class=64,batchSize=10,useCuda=True):
        super(NetCrnn, self).__init__()
        self.num_class = num_class
        self.batch=batchSize
        self.useCuda=useCuda
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, 5, padding=(2, 2),stride=(1,1)),#第1个
                nn.MaxPool2d(2, 2),
                nn.ReLU())
        self.conv2=ResidualBlock(16,32,1,self.downSample(16,32,1))
        self.conv3=nn.Sequential(
			nn.Conv2d(32,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.MaxPool2d((2,1), (2,1)),		   
			nn.ReLU())
        self.conv4=ResidualBlock(32,32,2,self.downSample(32,32,2))
        self.conv5=nn.Sequential(
			nn.Conv2d(32,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.MaxPool2d((2,1), (2,1)),
			nn.ReLU())
        self.conv6=ResidualBlock(32,64,1,self.downSample(32,64,1))
        self.conv7=nn.Sequential(
			nn.Conv2d(64,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.MaxPool2d((2,2), (2,2)),
			nn.ReLU())
        self.conv8=ResidualBlock(64,128,2,self.downSample(64,128,2))
        self.num_layers=5 
        self.gru = nn.GRU(128, 128, self.num_layers, batch_first=True, bidirectional=True)
        self.Lstm=nn.LSTM(128,128,2,True,True,bidirectional=True)
        self.fc=nn.Linear(256,self.num_class)
        self.hid=self.init_hidden(self.batch)
    def downSample(self,i_channel,o_channel,stride=1):
             conv=nn.Sequential(
				nn.Conv2d(i_channel,o_channel,3,stride,1),
				nn.BatchNorm2d(o_channel),
				nn.ReLU())
             return conv   

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, 128))
        if(self.useCuda):
        	return h0.cuda()
        else:
        	return h0.cpu()

    def forward(self, x):
        conv1=self.conv1(x)
        conv2=self.conv2(conv1)
        conv3=self.conv3(conv2)
        conv4=self.conv4(conv3)
        conv5=self.conv5(conv4)
        conv6=self.conv6(conv5)
        conv7=self.conv7(conv6)
        conv8=self.conv8(conv7)
        Gru=conv8.squeeze(2)
        Gru=Gru.transpose(1,2)
        out,hid = self.Lstm(Gru)
        out=out.reshape(-1,256)
        fc=self.fc(out)
        return fc

class NetCrnnOut(nn.Module):
    def __init__(self, num_class=64,batchSize=10,useCuda=True):
        super(NetCrnnOut, self).__init__()
        self.num_class = num_class
        self.batch=batchSize
        self.useCuda=useCuda
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, 5, padding=(2, 2),stride=(1,1)),#第1个
                nn.MaxPool2d(2, 2),
                nn.ReLU())
        self.conv2=ResidualBlock(16,32,1,self.downSample(16,32,1))
        self.conv3=nn.Sequential(
			nn.Conv2d(32,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.MaxPool2d((2,1), (2,1)),		   
			nn.ReLU())
        self.conv4=ResidualBlock(32,32,2,self.downSample(32,32,2))
        self.conv5=nn.Sequential(
			nn.Conv2d(32,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.MaxPool2d((2,1), (2,1)),
			nn.ReLU())
        self.conv6=ResidualBlock(32,64,1,self.downSample(32,64,1))
        self.conv7=nn.Sequential(
			nn.Conv2d(64,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.MaxPool2d((2,2), (2,2)),
			nn.ReLU())
        self.conv8=ResidualBlock(64,128,2,self.downSample(64,128,2))
        self.num_layers=5 
        self.gru = nn.GRU(128, 128, self.num_layers, batch_first=True, bidirectional=True)
        self.Lstm=nn.LSTM(128,128,2,True,True,bidirectional=True)
        self.fc=nn.Linear(256,self.num_class)
        self.hid=self.init_hidden(self.batch)
    def downSample(self,i_channel,o_channel,stride=1):
             conv=nn.Sequential(
				nn.Conv2d(i_channel,o_channel,3,stride,1),
				nn.BatchNorm2d(o_channel),
				nn.ReLU())
             return conv   

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, 128))
        if(self.useCuda):
        	return h0.cuda()
        else:
        	return h0.cpu()

    def forward(self, x):
        conv1=self.conv1(x)
        conv2=self.conv2(conv1)
        conv3=self.conv3(conv2)
        conv4=self.conv4(conv3)
        conv5=self.conv5(conv4)
        conv6=self.conv6(conv5)
        conv7=self.conv7(conv6)
        conv8=self.conv8(conv7)
        Gru=conv8.squeeze(2)
        Gru=Gru.transpose(1,2)
        out,hid = self.Lstm(Gru)
        out=out.reshape(-1,256)
        fc=self.fc(out)
        fc=torch.argmax(fc,dim=1)
        return fc




#alph="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,"
#c=len(alph)
net=NetCrnnOut(64,1,False).cuda()
net=net.cuda()
x=torch.randn(1,3,64,512,device='cuda')#[03457]
#label=torch.from_numpy(np.array([[0,3,4,5,7,7,9,3,1]])).cuda()
#input_length=torch.from_numpy(np.array([32])).cuda()
#target_length=torch.from_numpy(np.array([9])).cuda()
y=net(x)
#y=y.transpose(0,1)
#ctc_loss = nn.CTCLoss(blank=12, reduction='mean')
#loss=ctc_loss(y.log_softmax(2),label,input_length,target_length)


