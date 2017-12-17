import matplotlib.pyplot as plt
import tensorflow
from linearModels import readFile
from linearModels import saveFile
import glob
import tflearn
import numpy as np
import linearModels as lm

#train neural network for each item separately; neural network is implemented in tensorflow
def trainNN(trainX, trainY, cvX, cvY, file):
    g = tflearn.input_data(shape=[None, 13])
    g = tflearn.batch_normalization(g)
    g = tflearn.fully_connected(g, 512, 'relu')
    g = tflearn.batch_normalization(g)
    # g = tflearn.dropout(g, 0.7)
    g = tflearn.fully_connected(g, 256, 'relu')
    g = tflearn.batch_normalization(g)
    # g = tflearn.dropout(g, 0.7)
    g = tflearn.fully_connected(g, 128, 'relu')
    g = tflearn.batch_normalization(g)
    # g = tflearn.dropout(g, 0.7)
    g = tflearn.fully_connected(g, 1, activation='linear')
    g = tflearn.regression(g, optimizer='adam', loss='mean_square', learning_rate=0.0003)
    model = tflearn.DNN(g)
    model.fit(trainX, trainY, validation_set=(cvX, cvY), n_epoch=50)
    model.save(file)
    return model

def testModel(model, testX, fileName):
    testY = model.predict(testX)
    saveFile(fileName, testY)
    return testY

def main():
    for file in glob.glob(lm.PATH + 'pca13Xcv*train.csv'):
        trainX = readFile(file.replace('Xcv', 'X'))
        trainY = readFile(file.replace('pca13Xcv', 'Y'))
        testX = readFile(file.replace('Xcv', 'X').replace('train', 'test'))
        cvX = readFile(file)
        cvY = readFile(file.replace('pca13X', 'Y'))
        model = trainNN(trainX, np.reshape(trainY, (len(trainY), 1)), cvX, np.reshape(cvY, (len(cvY), 1)), file.replace('pca13Xcv', 'nn3ReluModel'))
        modelResultsY = testModel(model, testX, file.replace('pca13Xcv', 'nn3ReluY').replace('train', 'test'))
        print(modelResultsY)

if __name__ == "__main__":
    main()