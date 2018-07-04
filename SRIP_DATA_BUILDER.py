import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

class DATA:
    def salt_pepper_noise(self,image):
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
              # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out

    def preprocess_image(self,img , blur_kernel = (3 , 3)):
            img = cv2.blur(img , blur_kernel)
            img = self.salt_pepper_noise(img) ## salt pepper noise
            img[img > 20] = img[img > 20] - 20 
            return img
    
    def patchify(self , img  , scale = 1):
            p = int(self.patch_size/scale)
            r = int(img.shape[0] / p)
            c = int(img.shape[1] / p)
            patch_list = []
            for R in range(0,r):
                for C in range(0,c):
                    patch_list.append(img[R*p:(R+1)*p,C*p:(C+1)*p])
            return patch_list,r,c
    
    def reconstruct(self , li ,r , c , scale=1):
            image = np.zeros((int(r*(self.patch_size/scale)),int(c*(self.patch_size/scale)) , 3))
            print(image.shape)
            i = 0
            p = int(self.patch_size / scale)
            for R in range(0,r):
                for C in range(0,c):
                    image[R*p:(R+1)*p,C*p:(C+1)*p] = li[i]
                    i = i+1
            return np.array(image, np.uint8)
                    
    def __init__(self , folder = './' , patch_size = 64 ):
            self.folder = folder
            self.patch_size = patch_size
            
            
    def save_data(self):
            np.save('training_patches_Y.npy'  , self.training_patches_Y )
            if self.patch_size == 64:
                np.save('training_patches_2x.npy' , self.training_patches_2x) 
            elif self.patch_size == 128:
                np.save('training_patches_4x.npy' , self.training_patches_4x) 
            elif self.patch_size == 256:
                np.save('training_patches_8x.npy' , self.training_patches_8x) 
            #np.save('training_noisy_patches_2x.npy' , self.training_noisy_patches_2x)
            #np.save('training_noisy_patches_4x.npy' , self.training_noisy_patches_4x)
            #np.save('training_noisy_patches_8x.npy' , self.training_noisy_patches_8x)
    
    def load_data(self , folder=''):
            self.training_patches_Y  = np.load( folder+'/'+'training_patches_Y.npy') 
            if self.patch_size == 64:
                self.training_patches_2x = np.load( folder+'/'+'training_patches_2x.npy') 
            elif self.patch_size == 128:
                self.training_patches_4x = np.load( folder+'/'+'training_patches_4x.npy') 
            elif self.patch_size == 256:
                self.training_patches_8x = np.load( folder+'/'+'training_patches_8x.npy') 
            # self.training_noisy_patches_2x = np.load( folder+'/'+'training_noisy_patches_2x.npy')
            # self.training_noisy_patches_4x = np.load( folder+'/'+'training_noisy_patches_4x.npy')
            # self.training_noisy_patches_8x = np.load( folder+'/'+'training_noisy_patches_8x.npy')
            print(folder , " extreacted in !!")
    
    def construct_list(self ):
            train_data_list = []
            train_noisy_data_list = []
            i = 1
            for root , subFolder , files in os.walk(self.folder):
                try:
                    for file in files:
                        if file.split('.')[1] in ('jpg','png','jpeg'):
                            img = cv2.imread(root+'/'+file)
                            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                            r = self.patch_size - img.shape[0] % self.patch_size
                            c = self.patch_size - img.shape[1] % self.patch_size
                            noisy_img = self.preprocess_image(img)
                            img = np.pad(img, [(0,r),(0,c),(0,0)] , 'constant')
                            noisy_img = np.pad(noisy_img, [(0,r),(0,c),(0,0)] , 'constant')
                            train_data_list.append(img)
                            train_noisy_data_list.append(noisy_img)
                            print('processed image: ',i)
                            i = i + 1
                            # if i > 1:
                            #     break
                except OSError as e:
                    print('failed at file: ',root+'/'+file,' with ', e)
                # if i > 1:
                #     break
            self.train_data_list = train_data_list
            self.train_noisy_data_list = train_noisy_data_list
            self.training_patches_Y=[]
            self.training_patches_2x=[]
            self.training_patches_4x=[]
            self.training_patches_8x=[]
            self.training_noisy_patches_2x=[]
            self.training_noisy_patches_4x=[]
            self.training_noisy_patches_8x=[]
            for i in range(len(train_data_list)):
                img = train_data_list[i]
                noisy_img = train_noisy_data_list[i]
                self.training_patches_Y  += self.patchify(img)[0]
                if self.patch_size == 64:
                    self.training_patches_2x += self.patchify(cv2.resize(img , (int(img.shape[1]/2),int(img.shape[0]/2)) ,interpolation=cv2.INTER_LINEAR) , scale=2)[0]
                elif self.patch_size == 128:
                    self.training_patches_4x += self.patchify(cv2.resize(img , (int(img.shape[1]/4),int(img.shape[0]/4)) ,interpolation=cv2.INTER_LINEAR) , scale=4)[0]
                elif self.patch_size == 256:
                    self.training_patches_8x += self.patchify(cv2.resize(img , (int(img.shape[1]/8),int(img.shape[0]/8)) ,interpolation=cv2.INTER_LINEAR) , scale=8)[0]
                #self.training_noisy_patches_2x += self.patchify(cv2.resize(noisy_img , (int(noisy_img.shape[1]/2),int(noisy_img.shape[0]/2)) ,interpolation=cv2.INTER_LINEAR) , scale=2)[0]
                #self.training_noisy_patches_4x += self.patchify(cv2.resize(noisy_img , (int(noisy_img.shape[1]/4),int(noisy_img.shape[0]/4)) ,interpolation=cv2.INTER_LINEAR) , scale=4)[0]
                #self.training_noisy_patches_8x += self.patchify(cv2.resize(noisy_img , (int(noisy_img.shape[1]/8),int(noisy_img.shape[0]/8)) ,interpolation=cv2.INTER_LINEAR) , scale=8)[0]
            self.training_patches_Y = np.array(self.training_patches_Y,np.uint8)
            if self.patch_size == 64:
                self.training_patches_2x = np.array(self.training_patches_2x,np.uint8)
            elif self.patch_size == 128:
                self.training_patches_4x = np.array(self.training_patches_4x,np.uint8)
            elif self.patch_size == 256:
                self.training_patches_8x = np.array(self.training_patches_8x,np.uint8)
            #self.training_noisy_patches_2x = np.array(self.training_noisy_patches_2x,np.uint8)
            #self.training_noisy_patches_4x = np.array(self.training_noisy_patches_4x,np.uint8)
            #self.training_noisy_patches_8x = np.array(self.training_noisy_patches_8x,np.uint8)

if __name__ == "__main__":
    import time
    s = time.time()
    d = DATA(folder='./BSDS200_Padded') ## Call for constructing the Object. Give the root directory where it can find the
    # images. It will walk and find everything
    d.construct_list() # only when you need to create everything a new
    print('amount of data:' , d.training_patches_Y.shape)
    d.save_data() # save that constructed list
    d.load_data()
    elapsed = s - time.time()
    print(elapsed // 60 , "minutes were consumed of my life to await its creaton !!!")
