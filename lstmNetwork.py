from __future__ import division, print_function, absolute_import
from linearModels import readFile
from linearModels import saveFile
import glob
import tflearn
import numpy as np

#changes data to create sequences of length 7 that are input in LSTM network
#saves reorganized data into files contianing 7 in name
def dataPreparation(fileName, dataX, dataY, dataCvX, dataCvY, dataTestX):
    # This function is not finished, because I realized I don't have appropriate data for LSTM model
    # I still uploaded it, because other functions are good and can be used
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
