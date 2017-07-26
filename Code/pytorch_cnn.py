
import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import numpy as np
from torchvision import transforms

img = Image.open("image_name.jpg")
# img = np.array(img)
# print type(img)
tt = transforms.ToTensor()
img = ag.Variable(tt(img)).view(1,-1,300,300)
print img

#model
c1 = nn.Conv2d(3, 12, 25, stride=1)
ap = nn.AvgPool2d(4, stride=4)
c2 = nn.Conv2d(12, 24, 25, stride=1)
ll1 = nn.Linear(2904, 50)
ll2 = nn.Linear(50,1)
sig = nn.Sigmoid()

optimizer = optim.SGD((c1.parameters()), lr=0.01)
criterion = nn.MSELoss()
for i in range(300):	
	optimizer.zero_grad()
	out = sig(ll2(ll1(ap(c2(ap(c1(img)))).view(1,-1))))
	# print out
	loss = criterion(out, ag.Variable(torch.FloatTensor([2])))
	loss.backward()
	optimizer.step()
	if i%50 == 0:
		print i
		print out
		print loss.data[0]

# for i in range(30000):
# 	if i%1000 == 0:
# 		print i
# 		img = Image.open("image_name.jpg")
# 		tt = transforms.ToTensor()
# 		img = ag.Variable(tt(img)).view(1,-1,300,300)
# 		print img

# 		c1 = nn.Conv2d(3, 12, 25, stride=1)
# 		ap = nn.AvgPool2d(4, stride=4)
# 		c2 = nn.Conv2d(12, 24, 25, stride=1)
# 		ll1 = nn.Linear(2904, 50)
# 		ll2 = nn.Linear(50,1)
# 		sig = nn.Sigmoid()

# 		out = sig(ll2(ll1(ap(c2(ap(c1(img)))).view(1,-1))))
# 		print out