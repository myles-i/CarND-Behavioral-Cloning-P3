'''
Much of this work was inspired by blog posts and code posted by Vivek Yadav:
- https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.iijuava2v
- https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.pjmpsx84p
'''
import numpy as np
import cv2
from collections import namedtuple


AugImg_Vals = namedtuple('AugImg_Vals', ['brightness_scale',
										'translation_pixels',
										'added_shadow',
										'hflipped',
										'vflipped',
										'rotation_angle',
										'shear_angles'])
class AugImg_Vals:
	def __init__(self):
		self.brightness_scale = 1
		self.translation_pixels = (0,0)
		self.added_shadow = False
		self.hflipped = False
		self.vflipped = False
		self.rotation_angle = 0
		self.shear_angles = (0,0)


def augment_image_brightness(image, brightness_range = (0.25,1.5), return_rand_param = False):
	'''
	Description: applies a random brightness to an image within a specified range
	inputs:
	- image: Image in the form of a numpy array in (row, column) format
	- trans_range: A tuple of size 2 that indicates the range of translations in (y,x)
	outputs: 
	- img: The translated image
	- trans_amount: The percentages by which the image has been translated in y and x
	'''
	image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	random_bright =np.random.uniform(brightness_range[0],brightness_range[1])
	image1[:,:,2] = np.uint8(np.clip(np.float32(image1[:,:,2])*random_bright,0,255)) # saturate at 255
	image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
	if return_rand_param:
		return image1, random_bright
	else:
		return image1

def trans_image(image, trans_range, return_rand_param = False):
	'''
	Description: applies a random translation to an image within a specified range of pixels
	inputs:
	- image: Image in the form of a numpy array in (row, column) format
	- trans_range: A tuple of size 2 that indicates the range of translations in (y,x)
	outputs: 
	- img: The translated image
	- trans_amount: The number of pixels by which the image has been translated in y and x
	'''
	rows, cols, ch = image.shape 
	tr_x = np.random.randint(-trans_range[1],trans_range[1]+1)
	tr_y = np.random.randint(-trans_range[0],trans_range[0]+1)
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
	if return_rand_param:
		return image_tr, (tr_y,tr_x)
	else:
		return image_tr

def add_random_shadow(image, shadow_intensity = 0.5):
	'''
	Description: adds a random shadow to the image
	inputs:
	- image: Image in the form of a numpy array in (row, column) format
	outputs: 
	- img: The image containing the random shadow
	'''
	shp = image.shape
	top_y = shp[1]*np.random.uniform()
	
	top_x = 0
	bot_x = shp[0]
	bot_y = shp[1]*np.random.uniform()
	image_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	shadow_mask = 0*image_hsv[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	#random_bright = .25+.7*np.random.uniform()
	if np.random.randint(2)==1:
		random_bright = shadow_intensity # maybe make this brightness parameterized
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			image_hsv[:,:,2][cond1] = image_hsv[:,:,2][cond1]*random_bright
		else:
			image_hsv[:,:,2][cond0] = image_hsv[:,:,2][cond0]*random_bright    
	image = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2RGB)
	return image


def hflip(img,probability=0.5, return_rand_param = False):
	'''
	Description: Flips the image about horizontal axis with a given probability
	inputs:
	- img: Image in the form of a numpy array in (row, column) format
	- probability: The probability that the image will be flipped horizontally (0 to 1)
	outputs: 
	- img: The flippped (or un-flipped) image
	- image_flipped: indicates if the image has been flipped
	'''
	flip_image = np.random.uniform()<probability
	if flip_image:
		img = cv2.flip(img,0)

	if return_rand_param:
		return img, flip_image
	else:
		return img


def vflip(img,probability=0.5, return_rand_param = False):
	'''
	Description: Flips the image about vertical axis with a given probability
	inputs:
	- img: Image in the form of a numpy array in (row, column) format
	- probability: The probability that the image will be flipped vertically (0 to 1)
	outputs: 
	- img: The flippped (or un-flipped) image
	- image_flipped: indicates if the image has been flipped
	'''
	flip_image = np.random.uniform()<probability
	if flip_image:
		img = cv2.flip(img,1)

	if return_rand_param:
		return img, flip_image
	else:
		return img

def rotate_image(img, rot_range=(-15.,15.),return_rand_param = False):
	'''
	Description: applies a random rotation to an image within a specified range of degrees
	inputs:
	- img: Image in the form of a numpy array in (row, column) format
	- rot_range: The range of random rotations in degrees draw from a uniform distribution
	outputs: 
	- img: The rotated image
	- ang_rot: Angle by which the image has been rotated
	'''
	ang_rot = np.random.uniform(rot_range[0],rot_range[1])
	rows, cols, ch = img.shape   
	Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
	img = cv2.warpAffine(img,Rot_M,(cols,rows))
	if return_rand_param:
		return img, ang_rot
	else:
		return img

def shear_image(image, shear_range = (-10,10),return_rand_param = False):
	'''
	Description: applies a random shear to an image within a specified range of degrees (y,x)
	inputs:
	https://cs.colorado.edu/~mcbryan/5229.03/mail/55.htm
	- img: Image in the form of a numpy array in (row, column) format
	- shear_range: The range of random shears in degrees draw from a uniform distribution
	outputs: 
	- img: The sheared image
	- ang_rot: Degrees by which the image has been sheared (y,x)
	'''
	row,col,ch = image.shape
	angleY = np.random.uniform(-shear_range[0],shear_range[0])
	shear_My = np.array([[1, np.tan(np.radians(angleY))],[0,1]])
	angleX = np.random.uniform(-shear_range[1],shear_range[1])
	shear_Mx = np.array([[1, 0],[np.tan(np.radians(angleX)),1]])
	shear_M = np.dot(shear_My,shear_Mx)
	affine_trans = np.zeros((2,3))
	affine_trans[0:2,0:2] = shear_M
	image = cv2.warpAffine(image,affine_trans,(col,row))
	if return_rand_param:
		return image, (angleY,angleX)
	else:
		return image


def augment_image(image, shear_range = (0,0), rot_range =(0,0),
				vflip_prob = 0, hflip_prob = 0, add_shadow = False, shadow_intensity = 0.5,
				trans_range = (0,0), brightness_range = (1,1), return_rand_param = False):
	

	AugImg_Values = AugImg_Vals()
	image, AugImg_Values.brightness_scale = augment_image_brightness(image, brightness_range = brightness_range, return_rand_param = True)
	image, AugImg_Values.shear_angles= shear_image(image, shear_range = shear_range,return_rand_param = True)
	image, AugImg_Values.rotation_angle= rotate_image(image, rot_range=rot_range,return_rand_param = True)
	image, AugImg_Values.translation_pixels = trans_image(image, trans_range, return_rand_param = True)
	image, AugImg_Values.vflipped = vflip(image,probability=vflip_prob, return_rand_param = True)
	image, AugImg_Values.hflipped = hflip(image,probability=hflip_prob, return_rand_param = True)

	if add_shadow:
		image = add_random_shadow(image, shadow_intensity=shadow_intensity)
		AugImg_Values.added_shadow = True

	if return_rand_param:
		return image, AugImg_Values
	else:
		return image


	