import numpy as np
# Import Data

##################
# Some Constants
##################
img_size = (160,320,3)

###############
# Import Data
###############
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import pandas as pd 
import math
csv_data = pd.read_csv('./driving_log.csv')
img_names = csv_data['center']
steering = csv_data['steering']


# only keep a portion of steering angles that are 0
k_lim = 5

# Preallocate
names_shp = img_names.shape
zero_y = sum(steering==0)
nonzero_y = sum(steering!=0)
total_y = nonzero_y + math.floor(zero_y/k_lim)
shp = (total_y,)

X_train = np.empty(shp+img_size)
y_train = np.empty(shp)

idx= 0
idx2 = 0
idx3 =0
idx4 = 0
k = 0
for i in range(names_shp[0]):
	if steering[i] ==0:
		k+=1

	if(k%k_lim==0 and k !=0) or (steering[i] !=0): #keep all nonzero steering angles, and one in k_lim zero steering angles
		img = load_img(img_names[i])
		img = img_to_array(img)
		X_train[idx] = img
		y_train[idx] = steering[i]
		idx+=1
print('Data Loaded...\n')
data_loaded = True
################
# Split off validation data
#################
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Split off Validation Data...\n')