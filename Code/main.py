import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import numpy as np
import model as md
import random
from collections import defaultdict
from torchvision import transforms
from constants import HAVE_CUDA, BUCKET_SIZE, EMBEDDING_DIM, SIDELENGTH

if HAVE_CUDA:
	import torch.cuda as cuda


itemsfile = open("item_ids.txt","r")
items = []
for l in itemsfile:
	items.append(l.split("\n")[0])
datafile = open("pairs.txt","r")	#Change this to train/test as the data is divided
data = []
i=0
pdata = defaultdict(list)
users_to_ix = {}
for l in datafile:
	u,v,t = l.split(" ||| ")
	pdata[u] = v.split(",")
	users_to_ix[str(u)] = i
	i+=1
	if i%BUCKET_SIZE == 0:
		data.append(pdata)
		pdata = defaultdict(list)
if len(pdata)>0:
	data.append(pdata)
# print len(data)
# for i in range(len(data)):
# 	print len(data[i])
# 	break


# Load the model if available
if os.path.isfile(os.getcwd()+"/Checkpoints/img_model"):
	img_model = torch.load(os.getcwd()+"/Checkpoints/img_model")
else:
	img_model = md.ExtractImageVectors(EMBEDDING_DIM)
# Load user vectors
if os.path.isfile(os.getcwd()+"/Checkpoints/user_vts"):
	user_vts = torch.load(os.getcwd()+"/Checkpoints/user_vts")
else:
	user_vts = nn.Embedding(len(users_to_ix),EMBEDDING_DIM)#,max_norm = 1.0)

# Load AutoEncoder
if os.path.isfile(os.getcwd()+"/Checkpoints/auto_encoder"):
	AE = torch.load(os.getcwd()+"/Checkpoints/auto_encoder")
else:
	AE = md.AutoEncoder()

while 1:
	# Selecting a data bucket
	b_no = random.randint(0, len(data)-1)
	b_no = 1
	# Optimizer
	optimizer = optim.SGD(AE.parameters(), lr=0.001)
	# Train autoencoder
	
	md.trainAE(items[0:100],AE,optimizer)
	# Train the current batch
	# md.trainmodel1(data[b_no],items,user_vts,users_to_ix ,img_model,optimizer)
	break
