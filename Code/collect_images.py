import requests

ifile = open("item_urls.txt","r")
ofile = open("outliers.txt","w")
i = 0
for l in ifile:
	i+=1
	if i%1000==0:
		print i
	ll = l
	if len(l.split(","))>2:
		ofile.write(ll)
		continue
	else:
		n,m = l.split(",")
	n = "images/"+n+".jpg"
	m = m[0:-1]
	# print m
	img_data = requests.get(m).content
	with open(n, 'wb') as handler:
	    handler.write(img_data)
ifile.close()