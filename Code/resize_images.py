import PIL
from PIL import Image
import os

dir_to_save = "Resize_images/"
for root, dirs, files in os.walk("images/", topdown=True):
	i=0
	for name in files:
		i+=1
		if i%100==0:
			print i
		img = Image.open(os.path.join(root, name))
		basewidth = 500
		baseheight = 500
		w,h = img.size
		wpercent = (basewidth / float(w))
		hpercent = (baseheight / float(h))
		if wpercent<hpercent:
			hsize = int((float(h) * float(wpercent)))
			img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
			offset = (0, ((baseheight - hsize) / 2))
		else:
			wsize = int((float(w) * float(hpercent)))
			img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
			offset = (((basewidth - wsize) / 2),0)
		back = Image.new("RGB", [basewidth,baseheight])
		img.convert('RGB')
		back.paste(img,offset)
		back.save(dir_to_save+name)
		
	print i