
# coding: utf-8

# In[ ]:


# Wilson Gip
# CS 4990 - Machine Learning
# 10/21/18
# Assignment 2


# In[20]:


# Import all necessary libraries
import numpy as np
import pandas as pd
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.models import Model


# In[3]:


# Dataset path for training, validation, and testing
train_path = 'data/train'
valid_path = 'data/validation'
test_path = 'data/test'

# Set the image size to 255 for the Generator
image_size = 256


# ## Setup the data generators 

# In[4]:


# Create the ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
#     rotation_range = 20,
    width_shift_range= 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
#     fill_mode='nearest'
)

# Validation generator only requires a rescaling
valid_datagen = ImageDataGenerator(rescale = 1./255)


# In[5]:


# The batch size for training, validation, and testing
train_batchsz = 10
valid_batchsz = 10
test_batchsz = 10


# In[7]:


# Create generator for training dataset
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    class_mode = 'categorical',
    batch_size = train_batchsz)

# Create generator for validation dataset
valid_gen = valid_datagen.flow_from_directory(
    valid_path,
    target_size = (image_size, image_size),
    class_mode = 'categorical',
    batch_size  = valid_batchsz)


# ## Build the CNN with Pretrained Model VGG16 

# In[10]:


# Load the VGG16 Model with imagenet weights
vgg16_model = applications.VGG16(
    weights = None, 
    include_top=False, # Do not include the top since we don't need 1000 classifications
    input_shape=(image_size, image_size, 3))


# In[11]:


vgg16_model.summary()


# In[12]:


# Freeze all layers
for layers in vgg16_model.layers:
    layers.trainable = False


# ## Create new model and add VGG16

# In[13]:


# Add the weights into the VGG16 model
vgg16_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[22]:


x = vgg16_model.output
x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)


# In[23]:


filepath = 'pretrained_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
callbacks_list = [checkpoint]


# In[24]:


model = Model(inputs=vgg16_model.input, outputs=predictions)


# In[ ]:


# # Create the model
# model = Sequential()

# # Add the VGG16 model into the newely created model
# model.add(vgg16_model)

# # Add a new layer with 2 outputs
# # We will be changing weights on this smaller network that is ontop of the VGG16 model
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))


# In[25]:


# The number of samples for training and validation dataset
train_size = 11340
valid_size = 2830


# ## Compile the model 

# In[27]:


#  Compile the model
# model.compile(
#     loss="categorical_crossentropy", 
#     optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),
#     metrics=["accuracy"])
model.compile(
    loss="categorical_crossentropy", 
    optimizer=Adam(),
    metrics=["accuracy"])


# ## Start Training 

# In[ ]:


# Start training the model
model.fit_generator(
    train_gen,
    steps_per_epoch = train_size // train_batchsz,
    validation_data = valid_gen,
    validation_steps = valid_size // valid_batchsz,
    epochs = 20)

# # Save the weights after finish training
# model.save_weights('Pretrained_SGD_weights.h5')

