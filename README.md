# RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras
Residual Dense Network for Super Resolution implementation in Keras  
Original Paper : <a href="https://arxiv.org/abs/1802.08797">RDNSR</a>  
<hr>

## Model 

<img src="https://i.imgur.com/N8rGCsf.png">

The standard model and the RDB blocks  

<img src="https://i.imgur.com/CxDOxAQ.png">  

for each conv block the channel count = 64 and filter is (3,3) unless mentiond by the network  
otherwise. Skip connections are actually concatenation , the idea roots from Dense Networks  
And plus signs means Local or Global  

## Observation  

1. The model is small considering you are using only 3 RDB blocks , in total there are only around 800000 parameters,  
2. The model can be made much deeper only it will increase the training time. Converging takes a lot of time.  
3. Also never go with the per patch PSNR, since reconstruction would reduce it. Per patch PSNR should be atleast 80  
in order to get good results i.e less noise ad more clarity in picture.  
4. Though the paper does not mentions you can either take the input layer off and have the model repeatedly  
    initialized to create progressive upscalling or just change the SubPixel upscaling 'r' value in the code to  
    get 4x and 8x zoom.  
5. Awesome edge recreation.  
6. Global + Local Residual Learning is an effective way of preserving gemmoteric features.

## Usage  

### Data  

In the script SRIP DATA BUILDER just point the below line to whichever folder you have your training images in 
```
d = DATA(folder='./BSDS200_Padded' , patch_size = int(scale * 32))
```
It will create few npy files which training data. The training_patch_Y.npy contains the original data in patches  
of 64x64 or 128x128 how you specify. In case the image provided for training is not of proper shape, the script will pad them by black borders and convert the image in patches and save them in the npy file. trainig_patches_2x contains the 2x  
downscaled patches and respcitvely 4x and 8x contains the same. The noisy one is in case if you want to have  
them trained on grainy and bad image sample for robust training. The lines for noisy are commented , just uncomment and maybe at them in if else.  

```
p , r, c = DATA.patchify(img  , scale = 1)
```
By default patch size is 64. So  
<b> img </b>: Input image of shape (H,W,C)  
<b> scale </b>: scale determines the size of patch = patch_size / scale  
<b> returns </b>: list of patches , number of rows , number of columns  

### Training  

Just run the main.py , there are a few arguments in below format  

```
(rajat-py3) sanchit@Alpha:~/rajatKb/SRResNet$ python main.py -h
usage: main.py [-h] [--to TRYOUT] [--ep EPOCHS] [--bs BATCH_SIZE]
               [--lr LEARNING_RATE] [--gpu GPU] [--chk CHK] [--sample SAMPLE]
               [--scale SCALE] [--data FOLDER] [--test_image TEST_IMAGE]
               [--test_only TEST_ONLY]

control RDNSR

optional arguments:
  -h, --help            show this help message and exit
  --to TRYOUT
  --ep EPOCHS
  --bs BATCH_SIZE
  --lr LEARNING_RATE
  --gpu GPU
  --chk CHK
  --sample SAMPLE
  --scale SCALE
  --data FOLDER
  --test_image TEST_IMAGE
  --test_only TEST_ONLY
```
The file will be looking for few images to generate test results after every try out run. So you can provide  
your own test image or use one provided.  

test_only activate when you dont need to train , so you just point to model checkpoint get the results.

### Dataset  

Date: 30 June 2018  
The model is currently running on BSD300 and within 200 epochs generates image that captures the geometric shapes  
really well. Though small details are still smudged much like a MSE loss function. Look in to the training section  
of the paper and results section for furthur data.

#### Update:
 Trained on DIV2K and it is given good results considering we don't zooming in pixel level, there are some pepper noises there. But cannot be seen untill 200% zoom

### Results  

2x results  

low resolution image  
<img src="https://i.imgur.com/XkVB0En.png">  

bicubic image  
<img src="https://i.imgur.com/jt3UOS6.png">  

generated image from RDN  
<img src="https://i.imgur.com/7sWR0WD.png">


### Thanks to:  

https://github.com/tetrachrome/subpixel , Tetrachrome for his subpixel convolution layer in Keras. Though there were bugs  
But got it working. Will read the ESPCN paper for better understanding.
