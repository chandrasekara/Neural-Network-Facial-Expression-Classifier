
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# 
# ## Capstone Project
# 
# ### Using Computer Vision for Facial Expression Classification
# 
# This project investigates the use of computer vision techniques to build a classifier that can identify which facial expression is present, in a picture of a human face.
# 
# A deep learning approach will be taken to this problem, investigating the use of convolutional neural networks (CNNs). 
# 
# A simple benchmark CNN will first be investigated, followed by a more complex and deep network.
# 
# In addition, the use of pre-existing architectures for the [ImageNet challenge](http://www.image-net.org/challenges/LSVRC/) will be explored to aid performance in the facial expression classification problem. 
# 
# 
# 
# 
# 

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint  
from utility_functions import *

from keras import losses
from keras import backend as K
from keras import applications
from keras import optimizers

# ### Reading the Data
# 
# The dataset can be obtained from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

# In[2]:

# Allow for use of a section of the dataset to avoid large processing times
use_section_data = True
smaller_section_data = False

if use_section_data:
    data = pd.read_csv("section_fer2013.csv")    
    if smaller_section_data:
        data = pd.read_csv("smaller_section_fer2013.csv")
else:
    data = pd.read_csv("fer2013.csv")

# Create separate data structures for the labels and the features    
labels = data['emotion']
features = data.drop(['emotion','Usage'], axis=1, inplace=False)


# ### Pre-Processing the Data
# 
# The raw data is in .csv format, with every pixel value listed in a single cell, for each image, separated by spaces.
# 
# Below, each pixel value is separated, to be an individual value in a Python list, within a Data Frame.
# 
# In addition, the data will be converted to a tensor format.

# In[3]:

list_of_pixels = []
# Extract the pixel values for each image in the .csv file and put them into a list
for i in features['pixels']:
    list_of_pixels.append(i.split())

features.drop('pixels',axis=1,inplace=True)
features['pixels'] = pd.Series(list_of_pixels)


# Of all of the data that has been imported, 20% will be used for training, 10% for validation, and 70% for training.

# In[4]:

from sklearn.model_selection import train_test_split

# Split the total data into three sections: 80% training, 10% validation, and 20% testing
X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size = 0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.125, random_state=0)


# ### Sample Visualization
# 
# A few of the images from the dataset will be plotted below to ensure the data has been correctly processed.

# In[5]:

# Take a few sample images to provide a visualization
samples = dataframe_to_nparray(features,48)

# ### Frequency of Expressions
# 
# The distribution of how many times each expression appears in the dataset is important to inform what metrics are suitable for assessing the classifier. Below a bar graph showing the relative frequencies of each expression is shown.

# In[7]:

frequency_list = np.zeros(7)
for label in labels:
    frequency_list[int(label)] += 1
    
# Parse the numbers that correspond to different emotions in the dataset
emotion_indices = range(1,8)
x_axis_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise', 'Neutral']


# In[8]:

train_tensors = dataframe_to_nparray(X_train,48).astype('float32')/255
valid_tensors = dataframe_to_nparray(X_val,48).astype('float32')/255
test_tensors = dataframe_to_nparray(X_test,48).astype('float32')/255

train_targets = np.array(pd.get_dummies(y_train))
valid_targets = np.array(pd.get_dummies(y_val))
test_targets = np.array(pd.get_dummies(y_test))


# In[9]:

# For the fer2013 dataset obtained from Kaggle, each picture is a uniform size of 48x48
img_width = 48
img_height = 48


# ### Evaluation Metrics
# 
# Due to the uneven distribution of facial expressions in the dataset, illustrated above, accuracy alone is not a suitable measure of classifier performance. Instead <b>categorical cross entropy loss</b> will be used to rank classifiers.
# 

# ### Baseline Model
# 
# First, a simple CNN will be used to build a classifier, to which others can be compared. The baseline comprises only of one convolutional layer with 16 filters and is expected to pick up basic features in the faces that will give it some degree of accuracy with the classification task
# 

# In[10]:

model_baseline = Sequential()
model_baseline.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(48,48,1)))
model_baseline.add(MaxPooling2D(pool_size=2))
model_baseline.add(Flatten())
model_baseline.add(Dense(100, activation='relu'))
model_baseline.add(Dense(7, activation='softmax'))

model_baseline.summary()


# In[11]:

model_baseline.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:

epochs = 20
# Only use the model weights in the epoch with the least validation loss
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.baseline.hdf5', 
                               verbose=1, save_best_only=True)
history_baseline = model_baseline.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# In[13]:

model_baseline.load_weights('saved_models/weights.best.baseline.hdf5')


# In[14]:

print("Test accuracy: %.4f%% " % get_model_accuracy(model_baseline, test_tensors, test_targets))


# In[16]:

predictions_np = get_predictions_tensor(model_baseline, test_tensors)
print("Average categorical cross entropy: ", manual_total_cce(predictions_np, test_targets))


# ### Advanced Model

# Below, a more complex CNN architecture will be investigated. This model contains a higher number of convolutional layers and filters. It is expected this architecture will pick up more complex features in the image that will provide useful for classifying what the correct facial expression is.

# ### Image Augmentation
# 
# 
# 

# A high performance classifier for this task needs to have both rotational and translational invariance. To achieve this, image augmentation will be used, whereby images used for training are flipped and shifted randomly.

# In[27]:

from keras.preprocessing.image import ImageDataGenerator

def augment_data(train_tensors_input, valid_tensors_input, shift_width=0.1, shift_height=0.1, flip_hor=True,
                flip_vert=False):
    
    # Specify the ways in which image augmentation will be performed
    datagen_train = ImageDataGenerator(
        width_shift_range = shift_width,
        height_shift_range = shift_height,
        horizontal_flip = flip_hor,
        vertical_flip = flip_vert )

    datagen_valid = ImageDataGenerator(
        width_shift_range = shift_width,
        height_shift_range = shift_height,
        horizontal_flip = flip_hor,
        vertical_flip = flip_vert )

    datagen_train.fit(train_tensors_input)
    datagen_valid.fit(valid_tensors_input)
    
    return (datagen_train, datagen_valid)  


# In[28]:

(datagen_train, datagen_valid) = augment_data(train_tensors, valid_tensors)


# In[29]:

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()


# In[30]:

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[31]:

epochs = 50
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch_augmentation.hdf5', 
                               verbose=1, save_best_only=True)
#Training using image augmentation
batch_size = 20
history_advanced = model.fit_generator(datagen_train.flow(train_tensors, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_tensors.shape[0] // batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpointer],
                    validation_data=datagen_valid.flow(valid_tensors, valid_targets, batch_size=batch_size),
                    validation_steps=valid_tensors.shape[0] // batch_size)


# In[32]:

model.load_weights('saved_models/weights.best.from_scratch_augmentation.hdf5')


# In[33]:

# get index of predicted facial expression for test set images
print("Test accuracy: %.4f%% " % get_model_accuracy(model, test_tensors, test_targets))

# In[36]:

predictions_np = get_predictions_tensor(model, test_tensors)
print("Average categorical cross entropy: ", manual_total_cce(predictions_np, test_targets))


# ### Transfer Learning
# 
# It is suitable to explore the use of other CNN architectures that have been used for other computer vision applications. Below, a transfer learning approach using the VGG-19 architecture will be explored.

# In[18]:

# Use the same splits of data for the transfer learning approach
X_train_transf, X_test_transf, X_val_transf = X_train, X_test, X_val
y_train_transf, y_test_transf, y_val_transf = y_train, y_test, y_val
sample = dataframe_to_nparray(X_train_transf, 48, reshape=False)


# In[19]:

# Creating transfer tensors, which will have a value for each RGB channel

train_tensors_transf = dataframe_to_nparray(X_train_transf,48,reshape=False, triple_channels=True).astype('float32')/255
valid_tensors_transf = dataframe_to_nparray(X_val_transf,48, reshape=False, triple_channels=True).astype('float32')/255
test_tensors_transf = dataframe_to_nparray(X_test_transf,48, reshape=False, triple_channels=True).astype('float32')/255

train_targets_transf = np.array(pd.get_dummies(y_train_transf))
valid_targets_transf = np.array(pd.get_dummies(y_val_transf))
test_targets_transf = np.array(pd.get_dummies(y_test_transf))


# In[39]:

img_width, img_height = 48, 48
batch_size = 16
epochs = 50
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

model.summary()


# In[40]:

# Image augmentation for transfer model

from keras.preprocessing.image import ImageDataGenerator

datagen_train_transf = ImageDataGenerator(
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True) 

datagen_valid_transf = ImageDataGenerator(
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True) 

datagen_train_transf.fit(train_tensors_transf)
datagen_valid_transf.fit(valid_tensors_transf)


# In[41]:

# Change this code to freeze some of the layers of the VGG-19 model with the pre-trained weights
#for layer in model.layers:
#    layer.trainable = False


# In[42]:

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)
# creating the final model 
model_final = Model(input = model.input, output = predictions)
# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_final.summary()


# In[43]:

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.transfer_example_six.hdf5', 
                               verbose=1, save_best_only=True)
history_transfer = model_final.fit_generator(datagen_train_transf.flow(train_tensors_transf, train_targets_transf, batch_size=batch_size),
                    steps_per_epoch=train_tensors_transf.shape[0] // batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpointer],
                    validation_data=datagen_valid_transf.flow(valid_tensors_transf, valid_targets_transf, batch_size=batch_size),
                    validation_steps=valid_tensors_transf.shape[0] // batch_size)


# In[44]:

model_final.load_weights('saved_models/weights.best.transfer_example_six.hdf5')

# In[46]:

print('Test accuracy: %.4f%%' % get_model_accuracy(model_final, test_tensors_transf, test_targets_transf))


# In[47]:

predictions_np = get_predictions_tensor(model_final, test_tensors_transf)
print("Average categorical cross entropy is: ", manual_total_cce(predictions_np, test_targets_transf))


# ## Robustness Analysis
# 
# Below, the robustness of the final VGG-19 transfer learning model will be investigated. To do so, random Gaussian noise will be added to a number of images, and the performance of the classifier will be compared with when clean images were used.

# In[21]:

def onehot_to_label(one_hot_array, labels_array):
    for i in range(len(one_hot_array)):
        if one_hot_array[i] == 1:
            break
    return labels_array[i]


# In[25]:

def add_noise(samples, number_of_images, std_dev=0.025, seed_number=0):
    
    # Examine each pixel value
    for i in range(number_of_images):
        for pixel_x in samples[seed_number+i]:
            for pixel_y in range(len(pixel_x)):
                # Generate random Gaussian noise
                noise_ = np.random.normal(0,std_dev,1)[0]
                for rgb_value in range(len(pixel_x[pixel_y])):
                    pixel_x[pixel_y][rgb_value] += noise_
                    # Normalize out-of-range values
                    if pixel_x[pixel_y][rgb_value] < 0:
                        pixel_x[pixel_y][rgb_value] = 0
                    if pixel_x[pixel_y][rgb_value] > 1:
                        pixel_x[pixel_y][rgb_value] = 1
    
    return samples

# In[57]:

# add the gaussian noise to each of the test_tensors_transf
test_tensors_transf_noise = np.copy(test_tensors_transf)
num_images = test_tensors_transf_noise.shape[0]
test_tensors_transf_noise = add_noise(test_tensors_transf_noise, num_images, std_dev = noise_std_dev)


# In[58]:

predictions_np = get_predictions_tensor(model_final, test_tensors_transf_noise)
print("Categorical Cross Entropy Loss with noise added: ", manual_total_cce(predictions_np, test_targets_transf))


# In[59]:

print("Accuracy with noise added: ", get_model_accuracy(model_final, test_tensors_transf_noise, test_targets_transf))

