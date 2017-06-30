import requests
import PIL
from PIL import Image
import os
# image_url = "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg"

# img_data = requests.get(image_url).content
# with open('images/image_name.jpg', 'wb') as handler:
#     handler.write(img_data)
# 
# ########################################################################################    
# code snippet for resizing an image
# 
basewidth = 3000
img = Image.open("image_name.jpg")
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save("reimage_name.jpg")
# 
# ########################################################################################
# code snippet to get the image sizes

# from PIL import Image
# im = Image.open("image_name.jpg")
# width, height = im.size
# print width,height
# 
# ###########################################################################################
# Code Snippet for getting the max image width and height

# i=0
# mh = 0
# mw = 0
# for root, dirs, files in os.walk("images/", topdown=True):
# 	for name in files:
# 		i+=1
# 		if i%100==0:
# 			print i
# 		im = Image.open(os.path.join(root, name))
# 		w,h = im.size
# 		if w>mw:
# 			mw = w
# 		if h>mh:
# 			mh = h
# 	print i,mw,mh
# 	
# 	
