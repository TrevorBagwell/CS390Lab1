
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import matplotlib as mpl

random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#================<Set Structures>====================#

# These are the algorithms we can use
ALGORITHMSET = [ "guesser", "tf_net", "tf_conv" ] 

# This the data sets we can use
DATASETSET = [ "mnist_d" , "mnist_f" , "cifar_10" , "cifar_100_f", "cifar_100_c"]

#==========================<Options>=========================#

# This sets the algorithm to the desired item from the ALGORITHMSET
ALGORITHM = ALGORITHMSET[2]

# This sets the dataset to the desired item from the DATASETSET
DATASET = DATASETSET[2]

# These are self commented. No comments are needed. Except this one to explain why
# no comments are needed.
useSavedNet = False
saveNet = False
useAugmentedData = True
useCheckpoints = True
useEarlyStopping = True



#===========================<Metadata>===================#
# These are the data parameters for the data given the current dataset
# These include the shapes and class count

if DATASET == "mnist_d":
    # Number of classes for the one-hot array
    NUM_CLASSES = 10
    # This is the image height of the images
    IH = 28
    # This is the image width of the images
    IW = 28
    # This is the image depth, or the amount of colors
    IZ = 1
    # This is the image size, or the pixel count of the images
    IS = 784
    # This will be the input shape given by the neural net
    if ALGORITHM == "tf_conv":
        inShape = (IH, IW, IZ )
    else:
        inShape = (IS, )


elif DATASET == "mnist_f":
    # Number of classes for the one-hot array
    NUM_CLASSES = 10
    # This is the image height of the images
    IH = 28
    # This is the image width of the images
    IW = 28
    # This is the image depth, or the amount of colors
    IZ = 1
    # This is the image size, or the pixel count of the images
    IS = 784
    # This will be the input shape given by the neural net
    if ALGORITHM == "tf_conv":
        inShape = (IH, IW, IZ )
    else:
        inShape = (IS, )


elif DATASET == "cifar_10":
    # Number of classes for the one-hot array
    NUM_CLASSES = 10
    # This is the image height of the images
    IH = 32
    # This is the image width of the images
    IW = 32
    # This is the image depth, or the amount of colors
    IZ = 3
    # This is the image size, or the pixel count of the images
    IS = 3072
    # This will be the input shape given by the neural net
    if ALGORITHM == "tf_conv":
        inShape = (IH, IW, IZ )
    else:
        inShape = (IS, )


elif DATASET == "cifar_100_f":
    # Number of classes for the one-hot array
    NUM_CLASSES = 100
    # This is the image height of the images
    IH = 32
    # This is the image width of the images
    IW = 32
    # This is the image depth, or the amount of colors
    IZ = 3
    # This is the image size, or the pixel count of the images
    IS = 3072
    # This will be the input shape given by the neural net
    if ALGORITHM == "tf_conv":
        inShape = (IH, IW, IZ )
    else:
        inShape = (IS, )



elif DATASET == "cifar_100_c":
    # Number of classes for the one-hot array
    NUM_CLASSES = 100
    # This is the image height of the images
    IH = 32
    # This is the image width of the images
    IW = 32
    # This is the image depth, or the amount of colors
    IZ = 3
    # This is the image size, or the pixel count of the images
    IS = 3072
    # This will be the input shape given by the neural net
    if ALGORITHM == "tf_conv":
        inShape = (IH, IW, IZ )
    else:
        inShape = (IS, )

# This is the MAXIMUM_COLOUR VALUE
MAX_COLOUR_VALUE = 255

# Has the nueron count for the given layers
NUERON_COUNT_PER_LAYER = 768

# We set the dropout rate to around 20% because that's a little justifiable
DROPOUT_RATE = 0.25




#=========================<Conversion Functions>================================
# Converts an array of values to a one hot array
def convertToOneHot(data):
    # Instantiate the answers as an array
    # Will be an array of One-Hot arrays
    ans = []

    # Randomly create entries for the array based off of randomness
    for entry in data:

        # Instantiate a One-Hot array
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Set one of it's values to 1
        pred[random.randint(0, 9)] = 1

        # Add the entry to the answers array
        ans.append(pred)

    # We return a copy of the array as the array will be
    # destroyed when we are done with it
    return np.array(ans)





# Converts a set of one hot arrays to value based
# arrays. So [0, 1, 0] gets converted to 1
def convertFromOneHot(data):
    # Instantiate the answers as an array
    # Will be an array of One-Hot arrays
    ans = []

    # Randomly create entries for the array based off of randomness
    for entry in data:
        value = -1

        # Convert an entry from a one hot to a number
        for i in range(entry.size):
            if entry[i] == 1:
                value = i
                break


        # Add the entry to the answers array
        ans.append(value)

    # We return a copy of the array as the array will be
    # destroyed when we are done with it
    return np.array(ans)




#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    
    # Makes an array for the answers
    ans = []
    
    # Go through each entry and randomly guess a correct statement
    for entry in xTest:
        
        # Makes an array of length NUM_CLASSES
        pred = [0] * NUM_CLASSES
        
        # Randomly sets one of the numbers in the one-hot array
        pred[random.randint(0, NUM_CLASSES-1)] = 1
        
        # Appends the guess to answers
        ans.append(pred)
    
    # Return a numpy equivalent of ans
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 10):

    #TODO: Write code to build and train your keras neural net.
        # Instantiates our model
        tfModel = tf.keras.Sequential()

        # Instantiates the lossType for the model
        lossType = tf.keras.losses.CategoricalCrossentropy()

        # This is the optimizer with which we will compile the nueral net
        opt = tf.train.AdamOptimizer()


        # This adds the first layer
        tfModel.add( keras.layers.Dense( NUERON_COUNT_PER_LAYER ,  input_shape = inShape , activation = tf.nn.relu ) )

        # Sprinkle a little dropout in there
        tfModel.add( keras.layers.Dropout( DROPOUT_RATE/2 ) ) 

        # This adds the final layer
        tfModel.add( keras.layers.Dense( NUM_CLASSES , activation = tf.nn.softmax ) )

        # Compile the nueral net
        tfModel.compile( optimizer = opt , loss = lossType , metrics = ['accuracy'] )

        # Trains the data based on the training data provided
        tfModel.fit( x , y, epochs = eps , batch_size = 32 )

        return tfModel


# Description:
# > This fully builds a convolutional neural net
# Input(s):
# > x -
# > y -
# > eps -
# > dropout - 
# > dropRate - 
# Output(s):
# > tfCNN - the conv net that has been fully built, compiled, then trained
def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
   
    if DATASET == "mnist_d" or DATASET == "mnist_f":
        #
        return buildCNNForMNIST(x, y, eps=10, dropout=False, dropRate=dropRate)
    
    elif DATASET == "cifar_10":
        return buildCNNForCIFAR10(x, y, eps=10, dropout=False, dropRate=dropRate)
    
    
    elif DATASET == "cifar_100_f" or DATASET == "cifar_100_c":
        return buildCNNForCIFAR100(x, y, eps=10, dropout=False, dropRate=dropRate)

    else:
        raise ValueError("Dataset not recognized.")



#=======================<Individual CNN's for the given datasets>=======================#
# Description:
# > This fully builds, compiles, then fits a convolutional neural net for the MNIST Datasets
# Input(s):
# > x - this is the input data for training
# > y - this is the expected output data for training
# > eps - this is the count of epochs
# > dropout - toggles using dropout
# > dropRate - this is the dropout rate that will be used for data dropout
# Output(s):
# > tfCNN - the conv net that has been fully built, compiled, then trained
def buildCNNForMNIST(x,y, eps = 10, dropout = True, dropRate = 0.2):
    
    # Grab a model from the keras
        tfCNN = keras.Sequential()
    
        # This specifies the loss type. Just MSE for higher dimesnsions
        lossType = keras.losses.categorical_crossentropy
    
        # Sets the optimizer that we will train with
        opt = tf.train.AdamOptimizer()



        # Add a first layer of 2 dimensions
        tfCNN.add(keras.layers.Conv2D(32, kernel_size = (3,3), activation = tf.nn.relu, input_shape = inShape) )

        # Adds the second layer of a 2 dimensional net
        tfCNN.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation = tf.nn.relu) )

        # Adds the pooling layer
        tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2)))



        # Flatten the net
        tfCNN.add(keras.layers.Flatten())

        # Add a single layer with a relu activation
        tfCNN.add(keras.layers.Dense(128, activation = tf.nn.relu) )
        
        # Be soft but not too soft on max
        tfCNN.add(keras.layers.Dense(NUM_CLASSES, activation = tf.nn.softmax))



        # Make a parrallel gpu model
        #tfCNN_parrallel = keras.utils.multi_gpu_model(tfCNN, gpus=2)

        # Compile the model
        #tfCNN_parrallel.compile(optimizer = opt, loss = lossType)
        tfCNN.compile(optimizer = opt, loss = lossType, metrics = ['accuracy'])
    
        # Fit the data
        if useAugmentedData == True:
            # Get the augemented data
            datagen = augmentData(x)

            # Fit it on the augmented data
            tfCNN.fit_generator(datagen.flow(x, y, batch_size=64),
                    steps_per_epoch=x.shape[0]*32, epochs=eps,
                    validation_data=(x,y))
        else:            
            #tfCNN_parrallel.fit(x,y, epochs = eps)
            tfCNN.fit(x,y, epochs = eps, batch_size=32)
        

        # Return the neural net
        return tfCNN


# Description:
# > This fully builds, compiles, then fits a convolutional neural net for the CIFAR-10 Dataset
# Input(s):
# > x - this is the input data for training
# > y - this is the expected output data for training
# > eps - this is the count of epochs
# > dropout - toggles using dropout
# > dropRate - this is the dropout rate that will be used for data dropout
# Output(s):
# > tfCNN - the conv net that has been fully built, compiled, then trained
def buildCNNForCIFAR10(x,y, eps = 10, dropout = True, dropRate = 0.2):
    # Grab a model from the keras
    tfCNN = keras.Sequential()
    
    # This specifies the loss type. Just MSE for higher dimesnsions
    lossType = keras.losses.categorical_crossentropy
    
    # Sets the optimizer that we will train with
    #opt = tf.train.AdamOptimizer()
    
    # We are gonna do a stochastic gradient decent for this one right
    learn_rate = 0.015
    opt = keras.optimizers.SGD( lr=learn_rate, momentum=0.9, decay=learn_rate/eps, nesterov=False)



    # Add a first layer of 2 dimensions
    tfCNN.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation = tf.nn.relu, input_shape = inShape, padding = 'valid') )

    # Adds some dropout to the mix
    if dropout == True:
        tfCNN.add(keras.layers.Dropout(dropRate))
    
    # Adds the second layer of a 2 dimensional net
    tfCNN.add(keras.layers.Conv2D(128, kernel_size = (3,3), activation = tf.nn.relu, padding = 'valid') )

    # Adds the pooling layer
    tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

    # Adds the third layer of a 2 dimensional net
    tfCNN.add(keras.layers.Conv2D(128, kernel_size = (3,3), activation = tf.nn.relu, padding = 'valid') )

    # Adds the pooling layer
    tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2), padding = 'valid'))



    # Flatten the net
    tfCNN.add(keras.layers.Flatten())

    # Add a single layer with a relu activation
    tfCNN.add(keras.layers.Dense(512, activation = tf.nn.relu) )
    
    # Adds some dropout to the mix
    if dropout == True:
        tfCNN.add(keras.layers.Dropout(dropRate))

    # Adds a single layer with a relu activation
    tfCNN.add(keras.layers.Dense(256, activation = tf.nn.relu) )
    
    # Adds some dropout to the mix
    if dropout == True:
        tfCNN.add(keras.layers.Dropout(dropRate))

    # Be soft but not too soft on max
    tfCNN.add(keras.layers.Dense(NUM_CLASSES, activation = tf.nn.softmax))



    # Make a parrallel gpu model
    #tfCNN_parrallel = keras.utils.multi_gpu_model(tfCNN, gpus=2)

    # Compile the model
    #tfCNN_parrallel.compile(optimizer = opt, loss = lossType)
    tfCNN.compile(optimizer = opt, loss = lossType, metrics = ['accuracy'])
 
    # Fit the data
    #tfCNN_parrallel.fit(x,y, epochs = eps)
    #tfCNN.fit(x,y, epochs = eps, batch_size = 32)
    tfCNN = fitModel(x, y, tfCNN, eps = eps, batch_size = 32)

    return tfCNN


# Description:
# > This fully builds, compiles, then fits a convolutional neural net for the CIFAR-100 Datasets
# Input(s):
# > x - this is the input data for training
# > y - this is the expected output data for training
# > eps - this is the count of epochs
# > dropout - toggles using dropout
# > dropRate - this is the dropout rate that will be used for data dropout
# Output(s):
# > tfCNN - the conv net that has been fully built, compiled, then trained
def buildCNNForCIFAR100(x,y, eps = 4, dropout = True, dropRate = 0.2):
     # Grab a model from the keras
    tfCNN = keras.Sequential()
    
    # This specifies the loss type. Just MSE for higher dimesnsions
    lossType = keras.losses.categorical_crossentropy
    
    # Sets the optimizer that we will train with
    #opt = tf.train.AdamOptimizer()
    
    # We are gonna do a stochastic gradient decent for this one right
    learn_rate = 0.015
    opt = keras.optimizers.SGD( lr=learn_rate, momentum=0.9, decay=learn_rate/eps, nesterov=False)



    # Add a first layer of 2 dimensions
    tfCNN.add(keras.layers.Conv2D(216, kernel_size = (3,3), activation = tf.nn.relu, input_shape = inShape, padding = 'valid') )
    
    # Adds the pooling layer
    tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2), padding = 'valid' ))

    # Adds the normalization
    tfCNN.add(keras.layers.BatchNormalization())



    # Adds the second layer of a 2 dimensional net
    tfCNN.add(keras.layers.Conv2D(256, kernel_size = (3,3), activation = tf.nn.relu, padding = 'valid') )

    # Adds the pooling layer
    tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2), padding = 'valid' ))

    # Adds the normalization
    tfCNN.add(keras.layers.BatchNormalization())



    # Adds the third layer of a 2 dimensional net
    tfCNN.add(keras.layers.Conv2D(324, kernel_size = (3,3), activation = tf.nn.relu, padding = 'valid') )

    # Adds the pooling layer
    tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2), padding = 'valid' ))

    # Adds the normalization
    tfCNN.add(keras.layers.BatchNormalization())



    # Adds the third layer of a 2 dimensional net
    tfCNN.add(keras.layers.Conv2D(256, kernel_size = (3,3), activation = tf.nn.relu, padding = 'valid') )

    # Adds the pooling layer
    tfCNN.add(keras.layers.MaxPooling2D(pool_size = (2,2), padding = 'valid'))

    # Adds the normalization
    tfCNN.add(keras.layers.BatchNormalization())





    # Flatten the net
    tfCNN.add(keras.layers.Flatten())

    # Add a single layer with a relu activation
    tfCNN.add(keras.layers.Dense(2048, activation = tf.nn.relu) )
    
    # Adds some dropout to the mix
    if dropout == True:
        tfCNN.add(keras.layers.Dropout(dropRate))

    # Add a single layer with a relu activation
    tfCNN.add(keras.layers.Dense(2048, activation = tf.nn.relu) )
    
    # Adds some dropout to the mix
    if dropout == True:
        tfCNN.add(keras.layers.Dropout(dropRate))

    # Add a single layer with a relu activation
    tfCNN.add(keras.layers.Dense(512, activation = tf.nn.relu) )
    
    # Adds some dropout to the mix
    if dropout == True:
        tfCNN.add(keras.layers.Dropout(dropRate))

    # Be soft but not too soft on max
    tfCNN.add(keras.layers.Dense(NUM_CLASSES, activation = tf.nn.softmax))


    # Compile the model
    tfCNN.compile(optimizer = opt, loss = lossType, metrics = ['accuracy'])
 
    # Fit the data
    tfCNN = fitModel(x, y, tfCNN, eps = 1, batch_size = 32)

    return tfCNN

#TODO: Implement this function
def fitModel(x, y, model, eps = 16, batch_size = 32):

    # Instantiate a callback
    cb = keras.callbacks.Callback()

    # Instantiates the callback list and adds the callback to it
    cb_list = []
    cb_list.append(cb)
    
    # Instantiate the checkpoints
    if useCheckpoints == True:
        # This is the file path for the checkpoints
        filepath = getCheckpointsPath() + ".hdf5"
        
        # This is the actual checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        # Add it to the callbacks list
        cb_list.append(checkpoint)

    # Implements the earlystop
    es = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)

    cb_list.append(es)

    # Implements the training process with validation
    model.fit(x, y, epochs = eps, batch_size = batch_size, validation_split = 0.2, callbacks = cb_list)

    return model


#====================<Deep Memory Storage Functions>==================#

def getWeightsPath():
    # Should be in the savedNets folder, inside the folder with the same name as the dataset
    return "savedNetWeights/" + DATASET

def getCheckpointsPath():
    # Should be in the checkpoints folder, inside the folder with the same name as the dataset
    return "checkpoints/" + DATASET

def saveModelWeights(model, fileName = "kerasModel"):
    # Appends the h5 tag to the end of the file name
    fileName = fileName + ".h5"

    # Save the model to a local file
    model.save(fileName)

def loadModelWeights(fileName = "kerasModel"):
    # Loads an h5 file with the same name as the fileName
    fileName = fileName + ".h5"

    # Loads the model then returns it
    return keras.models.load_model(fileName)



#=========================<Data Augmentation Functions>==========================#

def augmentData(xTrain):
    # Make the datagenerator
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, featurewise_center=False,rotation_range=90, width_shift_range=0.05,height_shift_range=0.05, horizontal_flip=True)

    # Fit the data onto the datagenerator
    datagen.fit(xTrain)

    # Send back the datagen
    return datagen



#=========================<Pipeline Functions>==================================


# Gets the raw data from the datasets
# Returns a pair of pairs that contains the dataset
def getRawData():
    
    if DATASET == "mnist_d":
        
        # Get the mnist data
        mnist = tf.keras.datasets.mnist
        
        # Set the training and testing pairs to the dataset
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        
        

    elif DATASET == "mnist_f":
        
        # Get the fashion mnist data
        mnist = tf.keras.datasets.fashion_mnist
        
        # Set the training and testing pairs to the dataset
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        
    elif DATASET == "cifar_10":
        
        # Gets the data from the cifar10 data set
        cifar10 =  tf.keras.datasets.cifar10

        # Set the training and testing pairs to the dataset
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
        

    elif DATASET == "cifar_100_f":
        
        # Gets the data from the cifar100 data set
        cifar100 =  tf.keras.datasets.cifar100

        # Set the training and testing pairs to the dataset
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='fine')

        
    elif DATASET == "cifar_100_c":
        
        # Gets the data from the cifar10 data set
        cifar100 =  tf.keras.datasets.cifar100

        # Set the training and testing pairs to the dataset
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='coarse')


    else:
        
        raise ValueError("Dataset not recognized.")
    
    # Prints the dataset type
    print("Dataset: %s" % DATASET)
    
    # Prints the shape of the input training data
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    
    # Prints the shape of the expected output of the training data
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    
    # Prints the shape of the input of the testing data
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    
    # Prints the shape of the expected output of the testing data
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    
    # Returns the dataset
    return ((xTrain, yTrain), (xTest, yTest))


# Processes the data by:
#   1. Converting the input training data to streams of images
#   2. Converting all the colour schemes to floating point numbers between 
#      0.0-1.0 
#   3. Converting the output data to categorical groups
def preprocessData(raw):
    
    # Use these to handle the raw data
    ((xTrain, yTrain), (xTest, yTest)) = raw
    
    if ALGORITHM != "tf_conv":
        
        # Convert the shape of the training images to a datastream for each image
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        
        # Convert the shape of the testing images to a datastream for each image
        xTestP = xTest.reshape((xTest.shape[0], IS))
    
    else:
        
        # TODO: Figure out why the conv net needs all of these but the other neural nets don't
        # Each item will end up as an array of length IH*IW*IZ

        # Convert the shape of the training images to a datastream for each image
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        
        # Convert the shape of the testing images to a datastream for each image
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
        

    
    ## Change all the input data to values between 0.0-1.0

    # Update the training data's colour value
    xTrainP = xTrainP/MAX_COLOUR_VALUE

    # Update the testing data's colour value
    xTestP = xTestP/MAX_COLOUR_VALUE



    # Convert the input data to a categorical groupset
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    
    yTestP = to_categorical(yTest, NUM_CLASSES)
    
    
    
    
    # Prints out the shape of the new input training data
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    
    # Prints out the shape of the new input testing data
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    
    # Prints out the shape of the new output training data
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    
    # Prints out the shape of the new output testing data
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    
    # Returns the processed data
    return ((xTrainP, yTrainP), (xTestP, yTestP))




# This trains the model based on the training data
# Data should be (array<array<int>>, array<array<int>>) where
# the first array is of the datastream for each image and
# the second 
def trainModel(data):
    
    # This is the training data as it is needed
    xTrain, yTrain = data

    # Case handling for a guesser classifier
    if ALGORITHM == "guesser":

        # Guesser has no model, as it is just guessing.
        return None

        # ^^^^ It be like that sometimes
    
    # Case handling for an ANN
    elif ALGORITHM == "tf_net":
        
        # Print some stuff if we handle an ANN
        print("Building and training TF_NN.")
        
        # Return the model making function for the neural net
        return buildTFNeuralNet(xTrain, yTrain)
    
    # Case handling for a CNN
    elif ALGORITHM == "tf_conv":
        
        # Print some stuff, idk
        print("Building and training TF_CNN.")
        
        # Return a built model of the convolutional neural net
        if useSavedNet == False:
            return buildTFConvNet(xTrain, yTrain)

        else:
            return loadModelWeights(getWeightsPath())
    # Default case handling
    else:
        
        # When in doubt, throw an error
        raise ValueError("Algorithm not recognized.")



# Run it baby
def runModel(data, model):
    
    if ALGORITHM == "guesser":
        
        return guesserClassifier(data)
    
    elif ALGORITHM == "tf_net":
        
        print("Testing TF_NN.")
        
        preds = model.predict(data)
        
        for i in range(preds.shape[0]):
            
            oneHot = [0] * NUM_CLASSES
            
            oneHot[np.argmax(preds[i])] = 1
            
            preds[i] = oneHot
        
        return preds
    
    elif ALGORITHM == "tf_conv":
        
        print("Testing TF_CNN.")
        
        preds = model.predict(data)
        
        for i in range(preds.shape[0]):
            
            oneHot = [0] * NUM_CLASSES
            
            oneHot[np.argmax(preds[i])] = 1
        
            preds[i] = oneHot
        
        return preds
    
    else:
    
        raise ValueError("Algorithm not recognized.")


# This takes in a data set and the prediction set from the neural net
# 
def evalResults(data, preds):
    
    
    xTest, yTest = data
    
    acc = 0
    
    for i in range(preds.shape[0]):
        
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    
    accuracy = acc / preds.shape[0]
    
    print("Classifier algorithm: %s" % ALGORITHM)
    
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    
    print()

# Tests the saving and loading of a model
def saveAndLoadModel(model):

    # Creates the filename as a combination of the algorithm and the dataset 
    #fileName = ALGORITHM + "_" + DATASET
    fileName = getWeightsPath()

    # Saves the model
    saveModelWeights(model,fileName=fileName)

    # Loads the model
    loadedModel = loadModelWeights(fileName)

    return loadedModel



#=========================<Main>================================================

def init():
    
    # This gets the raw data for any of the nueral nets
    # Returns ((array<Matrix<Int>>, array<Int>), (array<Matrix<Int>>, array<Int>))
    #
    # The first pair is the training images and the training classifier
    # and the second pair is the testing images and the testing classifier
    raw = getRawData()
    
    # Processes the data to an ideal dataset for training
    # Returns ((array<array<Int>>, array<array<Int>>), (array<array<Int>>, array<array<Int>>))
    #
    # The first pair is the datastream of training images and the training output as One-Hot Arrays
    # and the second pair is the datastream of testing images and the testing output as One-Hot Arrays
    data = preprocessData(raw)
    
    # Just train the model of the dataset
    model = trainModel(data[0])
    
    # Returns the predictions 
    preds = runModel(data[1][0], model)
    
    print("\nEvaluation of the model before storage:")
    # Evaluates the results
    evalResults(data[1], preds)

    # Save and load the model
    if saveNet == True:
        # Call the save function
        model = saveAndLoadModel(model)

        print("\nEvaluation of the model after storage:")
        # Evaluates the results after the save and loading
        evalResults(data[1], preds)





def main():

    #for entry in DATASETSET:
        #DATASET = entry
        #init()
    DATASET = DATASETSET[1]
    init()

# This is the main function 
if __name__ == '__main__':
    
    # This just calls the main function at the gitgo
    main()
