# Wilson Gip
# CS 4990 - Machine Learning
# 10/21/18
# Assignment 2


# Import all necessary libraries
import numpy as np
import pandas as pd
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Dataset path for training, validation, and testing
train_path = 'data/train'
valid_path = 'data/validation'
test_path = 'data/testdata'

# Set the image size to 255 for the Generator
image_size = 256


# ## Setup the data generators 


# Create the ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range= 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    fill_mode='nearest')

# Validation generator only requires a rescaling
valid_datagen = ImageDataGenerator(rescale = 1./255)



# The batch size for training, validation, and testing
train_batchsz = 16
valid_batchsz = 8
test_batchsz = 10



# Read the submission csv file for the testing dataset filename
test_df = pd.read_csv('sample_submission.csv')


# Create generator for training dataset
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    classes=['female', 'male'],
    batch_size = train_batchsz,
    shuffle = True)

# Create generator for validation dataset
valid_gen = valid_datagen.flow_from_directory(
    valid_path,
    target_size = (image_size, image_size),
    classes     = ['female', 'male'],
    batch_size  = valid_batchsz,
    shuffle     = False)

# Use the validation generator to rescale the image of the testing dataset
test_gen = valid_datagen.flow_from_directory(
	test_path,
	batch_size = test_batchsz,
	target_size = (image_size, image_size),
	shuffle = False,
	class_mode = None)


# # Function to predict the testing dataset, returns the predictions
def predict(model_pred):
	test_gen.reset()
	print('Predicting test dataset')
	predictions = model_pred.predict_generator(
		test_gen,
		steps = 709,
		verbose = 0)
	print('Finish predicting!')
	return predictions


# # Output the predictions into a csv file for submission
def outputPrediction(predictions):
	test_gen.reset()
	predictions = predictions[:, 1:].flatten()
	test_filenames = test_gen.filenames
	print('prediction len', len(predictions))
	print('filename len', len(test_filenames))
	for i in range(len(test_filenames)):
		test_filenames[i] = test_filenames[i][5:]
	results = pd.DataFrame({
		"Id":test_filenames,
		"Expected": predictions
	})

	results.to_csv("submit.csv", index = False)
	print('Successfully created submit.csv file!')


# ## Build the CNN with Pretrained Model VGG16 


# Load the VGG16 Model with imagenet weights
vgg16_model = applications.VGG16(
    weights=None, 
    include_top=False, # Do not include the top since we don't need 1000 classifications
    input_shape=(image_size, image_size, 3))


vgg16_model.summary()


# Freeze all layers
for layers in vgg16_model.layers[:10]:
    layers.trainable = False
for layers in vgg16_model.layers:
    print(layers, layers.trainable)

# ## Create new model and add VGG16



# Add the weights into the VGG16 model
vgg16_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# Create the model
model = Sequential()

# Add the VGG16 model into the newely created model
model.add(vgg16_model)

# Add a new layer with 2 outputs
# We will be changing weights on this smaller network that is ontop of the VGG16 model
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4095, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()


# The number of samples for training and validation dataset
train_size = 22688
valid_size = 5672

# ## Compile the model 


early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5)
reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

#  Compile the model
model.compile(
    loss="categorical_crossentropy", 
    optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),
    metrics=["accuracy"])


# ## Start Training 


# Start training the model
model.fit_generator(
    train_gen,
    steps_per_epoch = train_size // train_batchsz,
    validation_data = valid_gen,
    validation_steps = valid_size // valid_batchsz,
    callbacks=[reducer, early],
    epochs = 50)

# Save the weights after finish training
model.save('Current_Model.h5')
print('Finished Training, saved model!')

predictions = predict(model)

outputPrediction(predictions)
