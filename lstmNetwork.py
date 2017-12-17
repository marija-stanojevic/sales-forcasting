from __future__ import division, print_function, absolute_import
from linearModels import readFile
from linearModels import saveFile
import glob
import tflearn
import numpy as np

#changes data to create sequences of length 7 that are input in LSTM network
#saves reorganized data into files contianing 7 in name
def dataPreparation(fileName, dataX, dataY, dataCvX, dataCvY, dataTestX):
    randomWeeks = readFile('randomWeeks.p')
    k = 1 # week in year
    l = 0 # current row in dataX
    trainX = [] # dimensions [?, 7, 120]
    trainY = [] # dimensions [?, 7, 1]
    cvX = [] # dimensions [?, 7, 120]
    cvY = [] # dimensions [?, 7, 1]
    testX = [] # dimensions [?, 7, 120]
    testLabel = [] # dimensions [?, 7, 1]
    for i in range(0, 240):
        for j in range(0, 7):
            if (dataX[l][4] != k):
                k = dataX[l][4]
                k += 1
                break
            elif (j == dataX[l][3]):
                trainX[i][j] = dataX[l] #TODO change this, because one product can be sold in different locations, we need to sum them all
                trainY[i][j] = dataY[l]
                l += 1
            elif (j < l):
                trainX[i][j] = 1 #TODO: make this a 120 list that has values for 3, 4, 5,
                trainX[i][j] = 0
    # TODO I didn't incorporte all the possibilities (weeks starting from Monday, Tuesday,...)
    # TODO input all dates starting from 29th December 2012 in trainX (as mean of previous and next date)
    # TODO input all missing dates in random weeks in cvX (as mean of previous and next date or one of them if there is no other)
    saveFile('X' + fileName, trainX)
    saveFile('Y' + fileName, trainY)
    saveFile('X' + fileName.replace('train', 'test'), testX)
    saveFile('label' + fileName.replace('train', 'test'), testLabel)
    saveFile('cvX' + fileName, cvX)
    saveFile('cvY' + fileName, cvY)
    return trainX, trainY, cvX, cvY, testX

#train neural network for each item separately; neural network is implemented in tensorflow
def trainLSTM(trainX, trainY, cvX, cvY, file):
    g = tflearn.input_data(shape=[None, 7, 13])
    g = tflearn.lstm(g, 256, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 128)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, 1, activation='linear')
    g = tflearn.regression(g, optimizer='sgd', loss='mean_square', learning_rate=0.001)
    model = tflearn.DNN(g)
    model.fit(trainX, trainY, validation_set=(cvX, cvY))
    model.save(file)
    return model

def testModel(model, testX, fileName):
    testY = model.predict(testX)
    saveFile(fileName, testY)
    return testY

def main():
    for file in glob.glob('pcaXcv*train.csv'):
        trainX = readFile(file.replace('Xcv', 'X'))
        trainY = readFile(file.replace('pcaXcv', 'Y'))
        testX = readFile(file.replace('Xcv', 'X').replace('train', 'test'))
        cvX = readFile(file)
        cvY = readFile(file.replace('pcaX', 'Y'))
        trainX, trainY, cvX, cvY, testX = dataPreparation(file.replace('pcaX', '7'), trainX, trainY, cvX, cvY, testX)
        model = trainLSTM(trainX, np.reshape(trainY, (len(trainY), 1)), cvX,  np.reshape(cvY, (len(cvY), 1)), file.replace('pcaXcv', 'nnModel'))
        modelResultsY = testModel(model, testX, file.replace('pcaX', 'lstmY').replace('train', 'test'))
        print(modelResultsY)

if __name__ == "__main__":
    main()