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
import time

if HAVE_CUDA:
	import torch.cuda as cuda


itemsfile = open("item_ids.txt","r")
items = []
nitems = []
itemsbin = []
i=0
for l in itemsfile:
	i+=1
	itm = l.split("\n")[0]
	items.append(itm)
	nitems.append(itm)
	if i%BUCKET_SIZE == 0:
		itemsbin.append(nitems)
		nitems = []
if len(nitems)>0:
	itemsbin.append(nitems)

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

if os.path.isfile(os.getcwd()+"/Checkpoints/optm"):
	optimizer = torch.load(os.getcwd()+"/Checkpoints/optm")
else:
	optimizer = optim.Adam(AE.parameters(), lr=0.001)

start_time = time.time()
iterind = 0
t_loss = 0
while 1:
	iterind+=1
	# print len(data)
	# print len(itemsbin)
	# Selecting a data bucket
	i_b_no = random.randint(0, len(itemsbin)-1)
	print iterind,i_b_no
	# b_no = 1
	# Optimizer
	
	# Train autoencoder
	
	t_loss += md.trainAE(itemsbin[i_b_no],AE,optimizer)
	# md.trainAE(itemsbin[b_no],AE,optimizer)

	torch.save(AE,os.getcwd()+"/Checkpoints/auto_encoder")
	# torch.save(optimizer,os.getcwd()+"/Checkpoints/optm")

	# Train the current batch
	# md.trainmodel1(data[b_no],items,user_vts,users_to_ix ,img_model,optimizer)
	if iterind%100 == 0:
		print "Time elapsed ======================== ",(time.time()-start_time)/60
		print "Total loss ========================== ",t_loss/100
		t_loss = 0
	# break
print (time.time()-start_time)/60