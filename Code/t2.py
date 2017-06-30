
"""
{"reviewerID": "A1KLRMWW2FWPL4", "asin": "0000031887", "reviewerName": "Amazon Customer \"cameramom\"", "helpful": [0, 0], "reviewText": "This is a great tutu and at a really great price. It doesn't look cheap at all. I'm so glad I looked on Amazon and found such an affordable tutu that isn't made poorly. A++", "overall": 5.0, "summary": "Great tutu-  not cheaply made", "unixReviewTime": 1297468800, "reviewTime": "02 12, 2011"}
{"reviewerID": "A2G5TCU2WDFZ65", "asin": "0000031887", "reviewerName": "Amazon Customer", "helpful": [0, 0], "reviewText": "I bought this for my 4 yr old daughter for dance class, she wore it today for the first time and the teacher thought it was adorable. I bought this to go with a light blue long sleeve leotard and was happy the colors matched up great. Price was very good too since some of these go for over $15.00 dollars.", "overall": 5.0, "summary": "Very Cute!!", "unixReviewTime": 1358553600, "reviewTime": "01 19, 2013"}
{"reviewerID": "A1RLQXYNCMWRWN", "asin": "0000031887", "reviewerName": "Carola", "helpful": [0, 0], "reviewText": "What can I say... my daughters have it in orange, black, white and pink and I am thinking to buy for they the fuccia one. It is a very good way for exalt a dancer outfit: great colors, comfortable, looks great, easy to wear, durables and little girls love it. I think it is a great buy for costumer and play too.", "overall": 5.0, "summary": "I have buy more than one", "unixReviewTime": 1357257600, "reviewTime": "01 4, 2013"}

"""

import json


jfile = open("reviews_Clothing_Shoes_and_Jewelry_5.json")
ifile = open("item_ids.txt","w")
x = set()
for f in jfile:
	jdata = json.loads(f)
	x.add(jdata["asin"])
# print jdata[0]
for i in x:
	ifile.write(i+"\n")
jfile.close()
ifile.close()