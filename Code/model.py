import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import numpy as np
import random
from torchvision import transforms
from constants import HAVE_CUDA, BUCKET_SIZE, EMBEDDING_DIM, SIDELENGTH

if HAVE_CUDA:
	import torch.cuda as cuda

class ExtractImageVectors(nn.Module):
 	"""docstring for ExtractImageVectors"""
 	def __init__(self,embedding_dim):
 		super(ExtractImageVectors, self).__init__()
 		self.c1 = nn.Conv2d(3, 12, 25, stride=1)
		self.ap = nn.AvgPool2d(6, stride=6)
		self.c2 = nn.Conv2d(12, 24, 25, stride=1)
		self.ll1 = nn.Linear(1944, embedding_dim)
		# self.ll2 = nn.Linear(50,1)
		# self.sig = nn.Sigmoid()
		# self.use_cuda = False

	def forward(self, input_img):
		out = self.ll1(self.ap(self.c2(self.ap(self.c1(input_img)))).view(1,-1))
		return out

class ScoreModel(nn.Module):
	"""docstring for ScoreModel"""
	def __init__(self):
		super(ScoreModel, self).__init__()
		# self.use_cuda = False

	def forward(self,uv,iv):
		out = torch.mm(uv.view(1,-1),iv.view(-1,1))
		return out

class CompareModel(nn.Module):
	"""docstring for CompareModel"""
	def __init__(self):
		super(CompareModel, self).__init__()
		# self.use_cuda = False
		self.sig = nn.Sigmoid()
		self.SM = ScoreModel()

	def forward(self,usr_vt,pitem,nitem):
		out = self.sig(self.SM(usr_vt,pitem)-self.SM(usr_vt,nitem))
		return out
		

def train(data, items, usr_vts, users_to_ix, model, optimizer, verbose=True):
	
	tot_loss = 0.0
	CM = CompareModel()
	tt = transforms.ToTensor()	#Helper class to convert Jpgs to tensors
	criterion = nn.MSELoss()
	if HAVE_CUDA:
		criterion.cuda()

	it = 0
	# model2 = ExtractImageVectors(EMBEDDING_DIM)
	# Pairwise Learning
	for usr in data:
		it+=1
		print it
		# Obtaining user vector
		usr_ix = users_to_ix[usr]
		uvt = usr_vts(ag.Variable(torch.LongTensor([usr_ix])))
		# print uvt.requires_grad

		# Clearing gradients
		optimizer.zero_grad()

		# Maximizing the observed items score compared to the unobserved
		for itm in data[usr]:

			# Using an unobserved item as a negative sample for pairwise learning
			unob_itm = itm
			while unob_itm in data[usr]:
				unob_itm = items[random.randint(0, len(items)-1)]

			# Loading the images from the filenames and converting them to FloatTensors to work with
			itm_img = Image.open(os.getcwd()+"/../Resize_images/"+itm+".jpg")			
			itm_img = ag.Variable(tt(itm_img)).view(1,-1,SIDELENGTH,SIDELENGTH)
			unob_itm_img = Image.open(os.getcwd()+"/../Resize_images/"+unob_itm+".jpg")			
			unob_itm_img = ag.Variable(tt(unob_itm_img)).view(1,-1,SIDELENGTH,SIDELENGTH)

			# Obtaining item Vectors
			pitem = model(itm_img)
			nitem = model(unob_itm_img)

			# Getting the prediction
			pred_out = CM(uvt,pitem,nitem)
			print pred_out

			# Calculating loss
			loss = 0
			loss += criterion(pred_out, ag.Variable(torch.FloatTensor([1])))
			tot_loss += loss.data[0]
			loss.backward(retain_variables=True)
			
			# print itm in data[usr]
			# print unob_itm in data[usr]

		# Back prop
		optimizer.step()
	print tot_loss