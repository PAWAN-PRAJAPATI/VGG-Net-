
from PIL import Image
import glob
import cv2
from tqdm import tqdm
from keras.utils import np_utils
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import pyxhook
from VGG_Net import VGG_11

'''
def kbevent(event):
    if event.Ascii == 32:
        exit(0)

hookman = pyxhook.HookManager()
hookman.KeyDown = kbevent
hookman.HookKeyboard()
hookman.start()

'''
HIGHT = 240
WIDTH = 140
n_class=5
MODEL_PATH = "MKeyNet_VGG11.h5"

l=[]
img_list = []

#Reading image from folder. Each folder represent each class and the are 5 folder as ntag_i'(i-(0-5))'
 
for i in tqdm(range(5)):
    path = "ntag_"+str(i)
    for filename in tqdm((glob.glob(path+'/*.jpg'))):
        img = cv2.imread(filename)
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(HIGHT,WIDTH))
        img_list.append(img)
    l.append(len(img_list))
print(l)

img_list = np.array(img_list)
img_list = img_list.astype('float32')
img_list /= 255
img_list = img_list.reshape(img_list.shape[0],3,HIGHT,WIDTH)

#Labels 
num_of_samples = img_list.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
labels[0:l[0]]=0
labels[l[0]:l[1]]=1
labels[l[1]:l[2]]=2
labels[l[2]:l[3]]=3
labels[l[3]:]=4

#Lable one hot encoding
Y = np_utils.to_categorical(labels, n_class)

#Call Model
model = VGG_11([HIGHT,WIDTH])
EPOCH = 50
hm_data = 2

for i in (range(EPOCH)):
    print(str(i)+"/"+str(EPOCH))
    x,y = shuffle(img_list,Y, random_state=64)
    for j in range(hm_data):   
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)
        model.fit(X_train,y_train,validation_data=(X_test,y_test),nb_epoch=1,verbose=1)

        print("Saving Model...")
        model.save(MODEL_PATH)
        print("Model Successfully Saved")

