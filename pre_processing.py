import cv2
from PIL import Image
import glob
from random import randint
import pandas as pd
import cv2
from tqdm import tqdm



n_class = 5
for j in tqdm(range(n_class)):
	img_lst =[]
	path ="n"+str(j)
	for filename in (glob.glob(path+'/*.jpg')):
		img_lst.append(filename)

	i=0
	for img_name in img_lst:
		l=[90,180]
		r= randint(0,1)
		i=i+1
		
		
		img = cv2.imread(img_name)
		img = cv2.resize(img, (240,240))
	   
		cv2.imwrite("ntag_"+str(j)+"/"+str(i)+".jpg",img) 
		
		#Rotating image randomly by 90 or 180 degree to increase dataset		
		M = cv2.getRotationMatrix2D((240/2,240/2),l[r],1)
		img = cv2.warpAffine(img,M,(240,240))
		i=i+1
		cv2.imwrite("ntag_"+str(j)+"/"+str(i)+".jpg",img) 
