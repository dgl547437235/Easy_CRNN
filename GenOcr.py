from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os

class GenOcrData:
	def __init__(self,alphabets,LabelTxt,SaveDir,StartIndex,RandomFontColor=False,RandomBgColor=False):
	    self.alphabets = alphabets
	    
	    self.useRandmoColor = RandomFontColor
	    self.defaultFontColor = (255,255,255)
	    self.defaultBgColor = (0,0,0,255)
	    self.LabelTxt = LabelTxt
	    self.SaveDir = SaveDir
	    self.StartIndex = StartIndex

	def GetRandomColor(self,IsBg):
		if(IsBg):
			return (random.randint(0,255),random.randint(0,255),random.randint(0,255),random.randint(0,255))
		else:
			return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
	def WriteLabel(self,PicPath,label):
		with open(self.LabelTxt,"a") as file:
			file.write(PicPath)
			file.write("--")
			file.write(label)
			file.write("\n")
			file.close()

	def GetOcrImg(self,text):
		length = len(text)
		if(self.useRandmoColor == False):
			new_img = Image.new('RGBA', (length * 100, 200), self.defaultBgColor)
		else:
			new_img = Image.new('RGBA', (length * 100, 200), self.GetRandomColor(True))

		draw = ImageDraw.Draw(new_img)		
		img_size = new_img.size
		font_size = 100
		fnt = ImageFont.truetype("arial.ttf", font_size)
		fnt_size = fnt.getsize(text)
		if(self.useRandmoColor == False):
			draw.text((10, 10), text, font=fnt, fill=self.defaultFontColor)
		else:
			draw.text((10, 10), text, font=fnt, fill=self.GetRandomColor(False))
		new_img = new_img.crop((10,15,12 + fnt_size[0],17 + fnt_size[1]))
		savePath = os.path.join(self.SaveDir,str(self.StartIndex) + ".png")
		new_img.save(savePath)
		self.WriteLabel(savePath,text)
		self.StartIndex = self.StartIndex + 1
		return
	def GetRandomText(self,minLength,maxLength):
		TextLength = random.randint(0,maxLength - minLength) + minLength

		TextStr = ""
		for i in range(TextLength):
			loc = random.randint(0,len(self.alphabets) - 1)
			a=alph[loc:loc + 1]
			if(a=="8*"):
			    TextStr = TextStr + " "
			else:
				TextStr=TextStr+a
		return TextStr
	def AutoGen(self,num,minLenght,maxLength):
		length = len(self.alphabets)
		
		for n in range(num):
			TextStr = self.GetRandomText(minLenght,maxLength)
			self.GetOcrImg(TextStr)

alph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,"
GOD = GenOcrData(alph,"label.txt",".//img//",0,False,False)
GOD.AutoGen(1000,3,15)




