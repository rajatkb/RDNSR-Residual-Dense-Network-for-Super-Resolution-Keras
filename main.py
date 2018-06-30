import cv2
import os

import argparse
parser = argparse.ArgumentParser(description='control SRCNN')
parser.add_argument('--to', action="store",dest="tryout", default=200)
parser.add_argument('--ep', action="store",dest="epochs", default=1000)
parser.add_argument('--bs', action="store",dest="batch_size", default=16)
parser.add_argument('--lr', action="store",dest="learning_rate", default=0.0001)
parser.add_argument('--gpu', action="store",dest="gpu", default=1)
parser.add_argument('--chk',action="store",dest="chk",default=-1)
# parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
# parser.add_argument('--scale', action='store' , dest = 'downscale' , default = 2)

values = parser.parse_args()
learning_rate = float(values.learning_rate)
batch_size = int(values.batch_size)
epochs = int(values.epochs)
tryout = int(values.tryout)
gpu=values.gpu
# test_sample = int(values.test_sample)
# downscale = int(values.downscale)
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
chk = int(values.chk)

import sys
import numpy as np
import matplotlib.pyplot as plt
from SRIP_DATA_BUILDER import DATA
from keras.models import Model
from keras.layers import Input,MaxPool2D,Deconvolution2D ,Convolution2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as k
import tensorflow as tf
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2

folder = '../BSDS200 processed'
CHANNEL = 3
DATA = DATA(folder = folder)

out_patch_size =  DATA.patch_size 
inp_patch_size = int(out_patch_size/ 2)
DATA.load_data(folder=folder)

print("Training data Y:" , DATA.training_patches_Y.shape)
print("Training data X:" , DATA.training_patches_2x.shape)


def PSNRLossnp(y_true,y_pred):
        return 10* np.log(255*2 / np.mean(np.square(y_pred - y_true)))

def SSIM( y_true,y_pred):
    u_true = k.mean(y_true)
    u_pred = k.mean(y_pred)
    var_true = k.var(y_true)
    var_pred = k.var(y_pred)
    std_true = k.sqrt(var_true)
    std_pred = k.sqrt(var_pred)
    c1 = k.square(0.01*7)
    c2 = k.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def PSNRLoss(y_true, y_pred):
        return 10* k.log(255**2 /k.mean(k.square(y_pred - y_true)))


class SRResnet:
    def L1_loss(self , y_true , y_pred):
        return k.mean(k.abs(y_true - y_pred))
    
    def RDBlocks(self,x,name):
            pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)
            out1 =  Concatenate(axis = self.channel_axis)([pass1,x]) # conctenated out put
            pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv2')(out1)
            out2 =  Concatenate(axis = self.channel_axis)([pass1,x,pass2])
            pass3 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv3')(out2)
            out3 = Concatenate(axis = self.channel_axis)([pass1,x,pass2,pass3])
            # feature extractor from the dense net
            feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out3)
            
            feat = Add()([feat , x])
            return feat
        
    def visualize(self):
            plot_model(self.model, to_file='model.png')
    
    def get_model(self):
        return self.model
    
    def __init__(self , channel = 3 , lr=0.0001 , patch_size=32):
            self.channel_axis = 3
            inp = Input(shape = (patch_size , patch_size , channel))

            pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)

            pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)

            RDB1 = self.RDBlocks(pass2 , 'RDB1')
            RDB2 = self.RDBlocks(RDB1 , 'RDB2')
            RDB3 = self.RDBlocks(RDB2 , 'RDB3')

            out = Concatenate(axis = self.channel_axis)([RDB1 , RDB2, RDB3])
            out = Convolution2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
            out = Convolution2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(out)

            output = Add()([out , pass1])

            output = Subpixel(3, (3,3), r = 2,padding='same',activation='relu')(output)
            output = Convolution2D(filters =3 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(output)

            model = Model(inputs=inp , outputs = output)
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr/100, amsgrad=False)
            model.compile(loss=self.L1_loss, optimizer=adam , metrics=[PSNRLoss,SSIM])
            if chk >=0 :
                print("loading existing weights !!!")
                model.load_weights('model_iter'+str(chk)+'.h5')
            self.model = model
            
    def fit(self , X , Y ,batch_size=16 , epoch = 1000 ):
            # with tf.device('/gpu:'+str(gpu)):    
            hist = self.model.fit(X, Y , batch_size = batch_size , verbose =1 , nb_epoch=epoch)
            return hist.history
    

net = SRResnet(lr = learning_rate)
net.visualize()
net.get_model().summary()

img = plt.imread('bsds_test2.jpg')
r = DATA.patch_size - img.shape[0] % DATA.patch_size
c = DATA.patch_size - img.shape[1] % DATA.patch_size
img = np.pad(img, [(0,r),(0,c),(0,0)] , 'constant')

Image.fromarray(img).save("padded_test.png")

lr_img = cv2.resize(img , (int(img.shape[1]/2),int(img.shape[0]/2)) ,cv2.INTER_CUBIC)

Image.fromarray(lr_img).save("padded_test_lr.png")

p , r , c = DATA.patchify(lr_img,scale=2) 

for i in range(chk+1,tryout):
    print("tryout no: ",i)
    net.fit(DATA.training_patches_2x , DATA.training_patches_Y , batch_size , epochs)
    net.get_model().save_weights('model_iter'+str(i)+'.h5')
    g = net.get_model().predict(np.array(p))
    gen = DATA.reconstruct(g , r , c , scale=1)
    gen[gen > 255] = 255
    gen[gen < 0] = 0
    Image.fromarray(gen).save("padded_gen_"+str(i)+".jpeg")
    print("Reconstruction Gain:", PSNRLossnp(img , gen))