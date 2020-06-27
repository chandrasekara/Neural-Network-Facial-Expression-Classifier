import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Written by Dhilan Chandrasekara, 2018


def dataframe_to_nparray(df_in, im_dim, reshape=True, triple_channels=False, horizontal_flip_double=False):
    
    # Convert a dataframe into a numpy array, suitably shaped for input to a CNN
    list_of_pixel_arrays = []  
    for image in df_in['pixels']:
        pixel_array = []
        for pixel in image:
            if not triple_channels:
                pixel_array.append(int(pixel))
            else:
                pixel_array.append([int(pixel),int(pixel),int(pixel)])
        # Organize the pixel values into a 2D array, of size length x width
        pixel_array = [pixel_array[x:x+im_dim] for x in range(0,len(pixel_array),im_dim)]
        list_of_pixel_arrays.append(pixel_array)
        
    if horizontal_flip_double == True:
        # Add in a copy of the image that is flipped horizontally
        reversed_array = []
        for pixel_row in pixel_array:
            reversed_array.append(pixel_row[::-1])
        list_of_pixel_arrays.append(reversed_array)
    np_array =  np.array(list_of_pixel_arrays)
    
    # Add an extra dimension for individual pixels, if needed to satisfy shape requirements for a model
    if reshape:
        return np_array.reshape(np_array.shape[0], 48, 48, 1)
    else:
        return np_array



def indiv_cce(predictions_vector, targets_vector, epsilon_ = 0.0001):
    
    # Calculates, for a single sample, the contribution to the total categorical cross entropy of the predictions with respect to the targets
    # Replaces zero with a small number to allow calculation of log
    for i in range(len(predictions_vector)):
        if predictions_vector[i] <= 0:
            predictions_vector[i] = epsilon_
    predictions = np.array(predictions_vector)
    targets = np.array(targets_vector)
    log_vector = -np.log(predictions_vector)
    
    return targets.dot(log_vector)



def manual_total_cce(predictions, targets ):
    
    # Finds the average categorical cross entropy of a set of predictions, using Numpy arrays instead of Tensorflow
    # Predictions and targets inputs must be numpy arrays with the shape: (number of samples, number of classes)
    total = 0
    for i in range(predictions.shape[0]):
        if predictions.shape[0] != targets.shape[0]:
            print("WRONG INPUT DIMENSIONS!")
            return None
        total += indiv_cce(predictions[i], targets[i])
    
    return float(total)/predictions.shape[0]



def get_model_predictions(input_model, input_test_tensors):
    
    # Evaluate what facial expressions the model predicts for each test image
    return [input_model.predict(np.expand_dims(tensor, axis=0)) for tensor in input_test_tensors]



def get_model_accuracy(input_model, input_test_tensors, input_test_targets):
    
    expression_predictions = [np.argmax(input_model.predict(np.expand_dims(tensor, axis=0))) for tensor in input_test_tensors]
    test_accuracy = 100*np.sum(np.array(expression_predictions)==np.argmax(input_test_targets, axis=1))/len(expression_predictions)
    
    return test_accuracy



def plot_model_history(history, save_file = None):

    # CODE OBTAINED FROM https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.title('Training and Validation data for Model')
    plt.xlabel('Training Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.show()
    
    #if save_file is not None:
    #    savefig(save_file)

    

def get_predictions_tensor(model, test_tensors):

    # Get model predictions and store them in an appropriately sized numpy array
    raw_predictions = get_model_predictions(model, test_tensors)
    dummy_array = np.array(raw_predictions)
    predictions_np = np.array(raw_predictions, dtype='float32').reshape(dummy_array.shape[0],7)
    
    return predictions_np