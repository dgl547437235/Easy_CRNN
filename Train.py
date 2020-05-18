import torch
import cv2 as  cv
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose, ToTensor
import math
import Model 
from torchvision import datasets, transforms
import torch.nn as nn
from torch.autograd import Variable
def GetData(labelText,StartIndex):
	imgPath=labelText.split("--")[0]
	label=labelText.split("--")[1]
	img=cv.imread(imgPath)
	hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
	h,s,v=cv.split(hsv)
	_,binary=cv.threshold(v,0,255,cv.THRESH_OTSU)
	hist = cv.calcHist([binary],[0],None,[256],[0,255])
	if(hist[0][0]<(binary.shape[0]*binary.shape[1]/2)):
		binary=cv.bitwise_not(binary)

	XMap=np.zeros(binary.shape[1])
	for i in range(binary.shape[1]):
		for j in range(binary.shape[0]):
			if(binary[j][i]==0):
				XMap[i]=XMap[i]+1
	loc=[]
	IsBegin=False
	start=0
	for i in range(0,XMap.shape[0]-1):
		if(XMap[i+1]<binary.shape[0] and XMap[i+1]<XMap[i] and IsBegin==False):#进入
			IsBegin=True
			start=i
		elif(XMap[i+1]>XMap[i] and XMap[i+1]==binary.shape[0] and IsBegin==True):#出去
			IsBegin=False
			loc.append([start,i])
	Imgs=[]
	for i in range(len(loc)):
		a=loc[i][0]
		b=loc[i][1]
		roi=binary[0:binary.shape[0],a:b]
		Imgs.append(roi)
	
	
def img_loader(img_path):
    img = cv.imread(img_path)
    img=cv.resize(img,(512,64))
    
    return img


class CaptchaData(Dataset):
    def __init__(self,LabelPath,alph,
                 transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.alph=alph
        self.Samples=self.ParaseLabelTxt(LabelPath)
        self.transform=Compose([ToTensor()])
    def __len__(self):

        return len(self.Samples)
    def ParaseLabelTxt(self,LabelPath):
        Samples=[]
        with open(LabelPath, 'r', encoding='utf-8') as f:
        	lines = f.readlines()    
        	for line in lines:
        		line=line.split("\n")[0]
        		imgPath=line.split("--")[0]
        		LabelStr=line.split("--")[-1]
        		LabelArray=self.ParaseLabelStr(LabelStr)
        		Samples.append([imgPath,LabelArray])
        return Samples
    def ParaseLabelStr(self,LabelStr):
        LabelArray=[]
        for c in LabelStr:
        	index=self.alph.find(c)
        	LabelArray.append(index)
        return LabelArray
    def __getitem__(self, index):
       
        imgPath=self.Samples[index][0]
        label=self.Samples[index][1]
        img=img_loader(imgPath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        Tl=len(label)
        #l=np.array((10))
        target=[0]*32
        for i in range(len(label)):
            target[i]=label[i]
        for i in range(len(label),32):
            target[i]=-1
        y_int = torch.LongTensor(target)
        #labelStr=imgPath.split("//")[-1].split(".")[0]

        return img,y_int,Tl,label

b=100
net=Model.NetCrnn(64,b).cuda()
ctc_loss = nn.CTCLoss(blank=63, reduction='mean')
alph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,"
DataSet=CaptchaData(".//label.txt",alph)
dataload=DataLoader(DataSet,b,transforms.Compose([transforms.ToTensor()]),drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)#定义优化器
input_length=torch.from_numpy(np.array([32]*b)).cuda()
net.load_state_dict(torch.load("crnn.pt"))
for epoch in range(10000):
	


	net.train()
	for img,label,target_length,_ in dataload:
		label = Variable(label[label > -1].contiguous()).cuda()
		
		
		target_length=torch.tensor(target_length).cuda()
		optimizer.zero_grad()
		y=net(img.cuda())
		y=y.view(-1,32,64)
		y=y.transpose(0,1)
		
		target_length=target_length.cuda()
		loss=ctc_loss(y.log_softmax(2),label,input_length,target_length)
		#optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if(epoch%50==0):
		net.eval()
		#test
		RightNum=0
		for img,label,target_length,labelStr in dataload:
			y=net(img.cuda())
			y=y.view(-1,32,64)
			y=torch.argmax(y,dim=2)
			for i in range(b):
				rawLabel=label[i]
				targetLabel=""
				for k in range(target_length[i]):
					targetLabel=targetLabel+str(alph[label[i][k].item():label[i][k].item()+1])
				strR=""
				temp=y[i]
				for j in range(0,y.shape[1]):
					v=y[i][j]
					v_=1
					if(j>0):
						v_=y[i][j-1]
					
					#arg_v_=torch.argmax(v_).item()
					if(v<63):
						if(j>0):
							if(v!=v_):
								strR=strR+str(alph[v:v+1])
						else:
							strR=strR+str(alph[v:v+1])
						
					
				if(strR==targetLabel):
					RightNum=RightNum+1
				#else:
				#	print(strR,"====",targetLabel)
		print("acc:",RightNum)
	if(epoch%100==0):
		print(loss)
	if(epoch%100==0):
		#print(loss)
		torch.save(net.state_dict(),"crnn.pt")
		model=Model.NetCrnn(64,1,False).cuda()

		model.load_state_dict(torch.load("crnn.pt"),strict=False)
		# An example input you would normally provide to your model's forward() method.
		#example = torch.rand(1, 3, 64, 512)
		# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
		#traced_script_module = torch.jit.trace(model, example)
		#traced_script_module.save("CrnnOut.pt")

		dummy_input = torch.randn(1, 3, 64,512, device='cuda') #定义输入的数据类型(B,C,H,W)为(10,3,224,224)
		output_names = [ "output" ]
		torch.onnx._export(model, dummy_input,"crnn.onnx",export_params=True)