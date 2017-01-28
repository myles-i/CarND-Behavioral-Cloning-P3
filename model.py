##################
# Preprocessing
##################
from keras.preprocessing.image import ImageDataGenerator

# Create an ImageData Generator that resclales the image from [0,1] and does some data augmentation
train_datagen = ImageDataGenerator(
        rotation_range=10, #degrees
        width_shift_range=0.0,
        height_shift_range=0.1,
        rescale=1./255, # rescale to between 0 and 1
        shear_range=0.1, # 2.8 degree sheer
        zoom_range=0.2,
        fill_mode='nearest')
# test datagen only rescale the test data images
batch_size = 128
test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow(
        X = X_train,
        y = y_train,
        batch_size=batch_size)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(
        X = X_val,
        y = y_val,
        batch_size=batch_size)

#######################
# Build network
#####################
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#Define convolutional layer
input_shape = (160,320,3)
dropout = 0.1 # percent that will dropout


model = Sequential()
####Convolution Layer 
model.add(Convolution2D(24, 5, 5,
						subsample = (2,2),
                        border_mode='valid',
                        input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 2
model.add(Convolution2D(36, 5, 5,
                        border_mode='valid',
                        subsample = (2,2)))
# model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 3
model.add(Convolution2D(48, 5, 5,
                        border_mode='valid',
                        subsample = (2,2)))
# model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 4
model.add(Convolution2D(64, 3, 3,
                        border_mode='valid',
                        subsample = (1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(dropout))
model.add(Activation('relu'))

####Convolution Layer 5
model.add(Convolution2D(64, 3, 3,
                        border_mode='valid',
                        subsample = (1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(dropout))
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
model.compile(loss='mean_squared_error', optimizer='Adam')
model.summary()



# conv_model = model
# model = Sequential()
# model.add(Flatten(input_shape=img_size))
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
nb_epoch = 6
history = model.fit(np.squeeze(X_train), y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data = (X_val,y_val))

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


