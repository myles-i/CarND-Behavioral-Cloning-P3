##################
# Load csv
##################
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import data_augmentation as da
import pandas as pd 
import cv2
csv_data = pd.read_csv('./driving_log.csv')
input_shape = (160,320,3)


##################
# Reserve some data for validation
##################
nb_val_samples = len(csv_data) *0.2 #reserve 20% for validation
val_idx = np.random.randint(len(csv_data), size = nb_val_samples)
val_idx = tuple(val_idx)


#######################
# Define data generators and processors
#######################
def grab_data(csv_data, index, camera):
    shift_val = 0.2
    if camera == 0: #left camera
        image= img_to_array(load_img(csv_data['left'][index].lstrip()))
        steering_cmd = csv_data['steering'][index] + shift_val# shift to simulate this as the center camera
    elif camera == 1: #center camera
        image= img_to_array(load_img(csv_data['center'][index].lstrip()))
        steering_cmd = csv_data['steering'][index]
    elif camera == 2: #right camera
        image= img_to_array(load_img(csv_data['right'][index].lstrip()))
        steering_cmd = csv_data['steering'][index] - shift_val # shift to simulate this as the center camer
    
    return image,steering_cmd


def augment_data(image, steering_cmd):
    # augment image (randomly flip about vertical axis, randomly translate image,
    # randomly change brightness, randomly add shadows)
    
    image, AugImg_Values = da.augment_image(image, shear_range = (0,0), rot_range =(0,0),
                                            vflip_prob = 0, hflip_prob = 0, add_shadow = False,
                                            trans_range = (20,50), brightness_range = (1,1), return_rand_param = True)
    # now augment steering_cmd based on how image was augmented
    steering_per_pixel = 0.002
    steering_cmd = steering_cmd + steering_per_pixel*AugImg_Values.translation_pixels[1]
    
    if AugImg_Values.vflipped:
        steering_cmd = -steering_cmd
    
    return image, steering_cmd

def my_train_datagen(csv_data,val_idxs, batch_size = 128):
    batch_X = np.zeros((batch_size,) + input_shape)
    batch_y = np.zeros(batch_size)
    m = 1
    while True:
        for i in range(batch_size):
            still_searching = True #initialize to less than 0.1
            while still_searching:
                #randomize which image is used, but not from validation set
                random_index = np.random.randint(len(csv_data))
                while val_idxs.__contains__(random_index):
                    random_index = np.random.randint(len(csv_data))
                #random_index = 57
                #randomize which camera to use
                random_camera = np.random.randint(3)
                #random_camera = 1

                image, steering_cmd = grab_data(csv_data,random_index,random_camera)
                image, steering_cmd = augment_data(image, steering_cmd)
                
                #keep no steering angles at first, then start increasingly using them to train 
                keep_prob = 0.5*(1- m)
                m = m*0.9999653432415313 # 0.5^(1/20000)=0.9999653432415313 (halves every 20000 images)
                small_steering_lim = 0.1
                if abs(steering_cmd)>small_steering_lim or np.random.uniform()<keep_prob:
                    still_searching = False

                #print(steering_cmd)


            batch_X[i] = image
            batch_y[i] = steering_cmd
        yield (batch_X, batch_y)

def my_validation_datagen(csv_data, val_idxs, batch_size = 128): 
    batch_X = np.zeros((batch_size,) + input_shape)
    batch_y = np.zeros(batch_size)
    counter = 0
    while True:
        for i in range(batch_size):
            counter+=1
            #cycle through validation data
            index = val_idxs[counter%len(val_idxs)]
            camera = 1 #always center camera for validation

            batch_X[i], batch_y[i] = grab_data(csv_data,index,camera)
        yield (batch_X, batch_y)
#######################
# Build network
#######################
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam

#Define convolutional layer
#dropout = 0# percent that will dropout


model = Sequential()
model.add(Lambda( lambda x: x/255.0-0.5, input_shape=input_shape))
####Convolution Layer 
model.add(Convolution2D(24, 5, 5,
						subsample = (2,2),
                        border_mode='valid'))
#model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 2
model.add(Convolution2D(36, 5, 5,
                        border_mode='valid',
                        subsample = (2,2)))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 3
model.add(Convolution2D(48, 5, 5,
                        border_mode='valid',
                        subsample = (2,2)))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 4
model.add(Convolution2D(64, 3, 3,
                        border_mode='valid',
                        subsample = (1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 5
model.add(Convolution2D(64, 3, 3,
                        border_mode='valid',
                        subsample = (1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(dropout))
model.add(Activation('relu'))

## Fully Connected Layers
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
Adamoptimizer = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=Adamoptimizer)
model.summary()



# conv_model = model
# model = Sequential()
# # model.add(Lambda( lambda x: x/255.0-0.5, input_shape=input_shape))
# model.add(Flatten(input_shape=input_shape))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='Adam', )

print('Model Compiled...\n')
##########
# Train
#########
print('Training...')
nb_epoch = 4
samples_per_epoch = 20000

training_gen = my_train_datagen(csv_data,val_idx, batch_size = 128)
validation_gen = my_validation_datagen(csv_data,val_idx, batch_size = 128)
history = model.fit_generator(training_gen, 
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                    verbose=1, validation_data = validation_gen, nb_val_samples=nb_val_samples)



##############
# Save to Disc
##############
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


