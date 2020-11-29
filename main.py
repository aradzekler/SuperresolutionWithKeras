import tensorflow as tf

import os
import math
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, BatchNormalization, Activation, Dense, Flatten, MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

from IPython.display import display


# TODO: need to download it to use locally
dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
root_dir = os.path.join(data_dir, "BSDS500/data")
valid_dir = os.path.join(data_dir, "BSDS500/data/images/val")

'''
preparing the dataset for later visuals
'''
dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")
valid_path = os.path.join(dataset, "val")


IMG_SIZE = 300
IMG_DIM = (300, 300)
upscale_factor = 3
input_size = IMG_SIZE // upscale_factor
BATCH_SIZE = 8
EPOCHS = 100


train_data_generator = ImageDataGenerator(
	rescale=1. / 255,  # maximum channels: 255
	rotation_range=30,
	shear_range=0.3,  # like tilting the image
	zoom_range=0.3,
	width_shift_range=0.4,  # off-centering the image
	height_shift_range=0.4,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode='nearest',
	validation_split=0.2)

train_ds = train_data_generator.flow_from_directory(
	root_dir,
	batch_size=BATCH_SIZE,
	subset="training",
	seed=1337,
	shuffle=True
)

# we dont need to fake images for validation..
test_data_generator = ImageDataGenerator(rescale=1. / 255)

valid_ds = train_data_generator.flow_from_directory(
	valid_path,
	color_mode='grayscale',
	target_size=IMG_DIM,
	batch_size=BATCH_SIZE,
	class_mode='categorical',
	subset='validation')



test_img_paths = sorted(
	[
		os.path.join(test_path, fname)
		for fname in os.listdir(test_path)
		if fname.endswith(".jpg")
	]
)

'''
processing image data, converting our images from 
the RGB color space to the YUV colour space.

For the input data (low-resolution images), we crop the image, 
retrieve the y channel (luminance), and resize it with the area 
method (use BICUBIC if you use PIL). We only consider the 
luminance channel in the YUV color space because humans are more 
sensitive to luminance change.

For the target data (high-resolution images), we just crop the
 image and retrieve the y channel.
'''


def process_input(input, input_size, upscale_factor):
	input = tf.image.rgb_to_yuv(input)
	last_dimension_axis = len(input.shape) - 1
	y, u, v = tf.split(input, 3, axis=last_dimension_axis)
	return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
	input = tf.image.rgb_to_yuv(input)
	last_dimension_axis = len(input.shape) - 1
	y, u, v = tf.split(input, 3, axis=last_dimension_axis)
	return y


train_ds = train_ds.map(
	lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

valid_ds = valid_ds.map(
	lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)


def model():
	_model = Sequential()

	# Block-1

	_model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
	                  input_shape=(IMG_SIZE, IMG_SIZE, 1)))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
	                  input_shape=(IMG_SIZE, IMG_SIZE, 1)))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block-2

	_model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block-3

	_model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block-4

	_model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())
	_model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block-5

	_model.add(Flatten())
	_model.add(Dense(64, kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())

	# Block-6

	_model.add(Dense(64, kernel_initializer='he_normal'))
	_model.add(Activation('elu'))
	_model.add(BatchNormalization())

	# Block-7

	_model.add(Dense(1, kernel_initializer='he_normal'))
	_model.add(Activation('softmax'))

	print(_model.summary())
	return _model


# will create a file checkpoint for our model, it will overwrite it every run until we will find the best model
checkpoint = ModelCheckpoint('MODEL.h5',
                             monitor='val_loss',  # monitor our progress by loss value.
                             mode='min',  # smaller loss is better, we try to minimize it.
                             save_best_only=True,
                             verbose=1)

# if our model accuracy (loss) is not improving over 3 epochs, stop the training, something is fishy
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          verbose=1,
                          restore_best_weights=True
                          )

# if our loss is not improving, try to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [checkpoint, earlystop, reduce_lr]

model = model()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

history = model.fit_generator(
	train_ds,
	steps_per_epoch=500 // BATCH_SIZE,
	epochs=EPOCHS,
	callbacks=callbacks,
	validation_data=valid_ds,
	validation_steps=100 // BATCH_SIZE)
