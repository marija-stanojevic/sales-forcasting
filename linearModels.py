from preprocessing import dataUnderstanding as du
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pickle
import csv
import datetime
import random
import glob
import math

PATH = '../../Desktop/Finished/6/'

# features: date(day in a week, week in a year, year), store_nbr(54), onpromotion, store_city(22), store_state(16), store_type(5), store_cluster(17), oil_price, holiday

# assigns value to holiday and days around holiday, h contains holiday type (Local - 3, Regional - 6, National - 9)
# i contains how many days away from the holiday we are and for each day lowers holiday value for 0.5; i is in [-4,4]
def getHolidayValue(h, i):
    i = i / 2.0
    if h[2] == 'Local':
        return 3 - i
    elif h[2] == 'Regional':
        return 6 - i
    else:
        return 9 - i

# assigns meaningful value to holiday days and days around holidays, so continual values can be used in regression
def prepareHolidayValuePerDay():
    holidays_events = du.readData('holidays_events.csv', 350, 6)
    holidays = {}
    for h in holidays_events:
        if h[5] == 'False' and h[0] >= '2013-01-01':
            hday = datetime.date(*map(int, h[0].split('-')))
            # gives value to a holiday day and 8 days around it based on holiday importance and closeness to holiday
            for i in range(-4, 5):
                hclose = (hday + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
                if (hclose in holidays):
                    holidays[hclose] += getHolidayValue(h, abs(i))
                else:
                    holidays[hclose] = getHolidayValue(h, abs(i))
    h = np.empty([len(holidays), 2], dtype=object)
    h[:, 0] = list(holidays.keys())
    h[:, 1] = list(holidays.values())
    return h

# Randomly chooses weeks that will be used for CV (5% of training data)
def randomWeeksDates(): # 2013:1-52, 2014:53-104, 2015:105-157, 2016:158-209, 2017:210-241
    dates = []
    randomWeeks = np.random.choice(241, (241 * 7) // 80)  # with this around 5% of data goes to CV
    saveFile('randomWeeks.p', randomWeeks)
    for week in randomWeeks:
        if week >= 210:
            year = 2017
            yearWeek = week - 209
        elif week >= 158:
            year = 2016
            yearWeek = week - 157
        elif week >= 105:
            year = 2015
            yearWeek = week - 104
        elif week >= 53:
            year = 2014
            yearWeek = week - 52
        else:
            year = 2013
            yearWeek = week
        randomWeekDay = random.randint(0, 6)
        d = datetime.datetime.strptime(str(year) + ' ' + str(yearWeek) + ' 0', '%Y %W %w') + datetime.timedelta(days=randomWeekDay)
        for i in range(0, 7):
            dates.append((d + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
    return dates

# transforms date feature into three features that I'll consider continuous
def dateToFeatures(dataX):
    for i in range(0, len(dataX)):
        year = int(dataX[i][0][0:4])
        month = int(dataX[i][0][5:7])
        day = int(dataX[i][0][8:10])
        date = datetime.date(year, month, day)
        weekDay = date.weekday()
        yearWeek = date.isocalendar()[1]
        dataX[i][9] = weekDay
        dataX[i][10] = yearWeek
        dataX[i][11] = year - 2013

# gives value to each city, state, type and cluster; number goes from 0 to len(object)
def getStoreInfoDictionaries():
    cities = {}
    states = {}
    types = {}
    clusters = {}
    with open('stores.csv', newline='') as f:
        reader = csv.reader(f)
        ci = 0
        si = 0
        ti = 0
        cli = 0
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] not in cities:
                    cities[row[1]] = ci
                    ci += 1
                if row[2] not in states:
                    states[row[2]] = si
                    si += 1
                if row[3] not in types:
                    types[row[3]] = ti
                    ti += 1
                if row[4] not in clusters:
                    clusters[row[4]] = cli
                    cli += 1
    return cities, states, types, clusters

# transform each multi-value categorical feature into multiple 0/1 value features
# reads files item by item and adds binary categorical features instead of multi-label categorical features
def decomposeCategoricalFeatures(dataX):
    cities, states, types, clusters = getStoreInfoDictionaries()
    for i in range(0, len(dataX)):
        featureNbr = int(dataX[i][1]) + 12 # store_nbr features
        dataX[i][featureNbr] = 1
        featureNbr = cities[dataX[i][2]] + 66 # store city features
        dataX[i][featureNbr] = 1
        featureNbr = states[dataX[i][3]] + 88 # store state features
        dataX[i][featureNbr] = 1
        featureNbr = types[dataX[i][4]] + 104 # store type features
        dataX[i][featureNbr] = 1
        featureNbr = clusters[dataX[i][5]] + 109  # store cluster features
        dataX[i][featureNbr] = 1

#normalize features to (-1, 1) range
def normalizeFeatures(dataX):
    for i in [6, 7, 9, 10, 11]:
        mean = np.mean(dataX[:, i])
        std = np.std(dataX[:, i])
        if std > 0 :
            dataX[:, i] = (dataX[:, i] - mean) / std
        else:
            dataX[:, i] -= mean
    return dataX

# reads files item by item and normalize sales for each file(item) separately
def normalizeSales(dataY):
    mean = np.mean(dataY)
    std = np.std(dataY)
    if std > 0:
        dataY = (dataY - mean) / std
    else:
        dataY -= mean
    return dataY, mean, std

# extracts data for cross validation based on days from randomly chosen weeks
def pullOutSamplesForCV(dataFile, trainX, trainY, dates):
    cvX = []
    cvY = []
    for i in range(0, len(trainX)):
        if trainX[i][0] in dates:
            cvX.append(trainX[i])
            cvY.append(trainY[i])
    saveTrainingData('cv' + dataFile, np.array(cvX), np.array(cvY), True)

def saveTrainingData(dataFile, dataX, dataY, training):
    if (len(dataX) > 0):
        dataX = dataX[:, 6:]
        with open('X' + dataFile, 'wb') as pickleFile:
            pickle.dump(dataX, pickleFile)
        if training:
            with open('Y' + dataFile, 'wb') as pickleFile:
                pickle.dump(dataY, pickleFile)
        else:
            with open('labels' + dataFile, 'wb') as pickleFile:
                pickle.dump(dataY, pickleFile)

def defineFeaturesAndSave(dataFile, data, holidays, oil, stores, training, itemNo, mean, std, dates):
    dataX = np.zeros([len(data), 126], dtype=object)
    dataY = np.zeros([len(data), ], dtype=object)
    # labels are ids in training/test files for which prediction has to be submitted
    # since I don't need them for training, but I need them for test data, so I'll save them instead of labels for test
    labels = np.zeros([len(data), ], dtype=object)
    dataX[:,[0,1]] = data[:,[1,2]]
    if training:
        dataY[:] = data[:, 4]
        dataX[:, 8] = (data[:, 5] == 'True').astype(int)
    else:
        labels[:] = data[:, 0]
        dataX[:, 8] = (data[:, 4] == 'True').astype(int)
    data = None
    j = 0
    k = 0
    for i in range(0, len(dataX)):
        dataX[i][2:6] = stores[int(dataX[i][1]) - 1][1:5]
        if (dataX[i][0] > oil[j][0]):
            j += 1
        l = 1
        while (oil[j][1] == ''):
            oil[j][1] = oil[j - l][1]
            l += 1
        dataX[i][6] = float(oil[j][1])
        while (dataX[i][0] > holidays[k][0]):
            k += 1
        if (dataX[i][0] == holidays[k][0]):  # otherwise, value will stay 0, since it's not holiday
            dataX[i][7] = holidays[k][1]
    dateToFeatures(dataX)
    dataX = normalizeFeatures(dataX)
    decomposeCategoricalFeatures(dataX)
    if training:
        dataY, mean[itemNo], std[itemNo] = normalizeSales(dataY.astype(np.float))
        pullOutSamplesForCV(str(itemNo) + dataFile, dataX, dataY, dates)
        saveTrainingData(str(itemNo) + dataFile, dataX, dataY, training)
    else:
        saveTrainingData(str(itemNo) + dataFile, dataX, labels, training)

# creates trainX/testX and trainY/testY from files received in competition
# trainX/testX has features listed on the top of this page
def prepareForTraining(dataFile, dataLen, training):
    mean = {}
    std = {}
    dates = randomWeeksDates()
    holidays = prepareHolidayValuePerDay()
    oil = du.readData('oil.csv', 1218, 2)
    stores = du.readData('stores.csv', 54, 5)
    if training:
        data = du.readData(dataFile, dataLen, 6)
    else:
        data = du.readData(dataFile, dataLen, 5)
    data = data[np.lexsort((data[:, 2].astype(int), data[:, 1], data[:, 3]))] #sort by storeNbr, date, itemNbr
    itemNo = data[0][3]
    dataX = []
    for i in range(0, len(data)):
        if itemNo != data[i][3]:
            if (len(dataX) > 0):
                defineFeaturesAndSave(dataFile, np.array(dataX), holidays, oil, stores, training, itemNo, mean, std, dates)
            dataX = []
            itemNo = data[i][3]
        dataX.append(data[i])
    if training:
        saveFile('Ymeans' + dataFile, mean)
        saveFile('Ystd' + dataFile, std)

#saves data into file
def saveFile(fileName, data):
    with open(fileName, 'wb') as pickleFile:
        pickle.dump(data, pickleFile)

# reads data from file
def readFile(fileName):
    data = pickle.load(open(fileName, 'rb'))
    return data

# checks variance of variables and reduce dimension (some dimensions are correlated, e.g. storeCity and storeState)
def pca(n_components):
    for cvFile in glob.glob('Xcv*train.csv'):
        file = cvFile.replace('Xcv', 'X')
        testFile = file.replace('Xcv', 'X').replace('train', 'test')
        trainX = readFile(file)
        cvX = readFile(cvFile)
        pca = PCA(n_components=n_components)  # all components that have variance more than 0.01
        pca.fit(trainX)
        trainX = pca.transform(trainX)
        cvX = pca.transform(cvX)
        saveFile('pca' + str(n_components) + file, trainX)
        saveFile('pca' + str(n_components) + cvFile, cvX)
        try:
            testX = readFile(testFile)
            testX = pca.transform(testX)
            saveFile('pca' + str(n_components) + testFile, testX)
        except FileNotFoundError:
            print(testFile + ' not found')

# fits linear regression model for each item(file) of training data
def linReg(trainX, trainY):
    reg = LinearRegression()
    reg.fit(trainX, trainY)
    return reg

# fits lasso regression model for each item(file) of training data
def lassoReg(trainX, trainY, alpha):
    lasso = Lasso(alpha)
    lasso.fit(trainX, trainY)
    return lasso

# fits ridge regression model for each item(file) of training data
def ridgeReg(trainX, trainY, alpha):
    ridge = Ridge(alpha)
    ridge.fit(trainX, trainY)
    return ridge

# fits SVM regression for each item(file) of training data
def svrLinear(trainX, trainY, C, epsilon):
    svr = SVR(C=C, epsilon=epsilon, kernel='linear')
    svr.fit(trainX, trainY)
    return svr

# fits SVM regression for each item(file) of training data
def svrGaussian(trainX, trainY, C, epsilon):
    svr = SVR(C=C, epsilon=epsilon)
    svr.fit(trainX, trainY)
    return svr

def crossValidateModel(model, dataX, dataY):
    predY = model.predict(dataX)
    return mean_squared_error(dataY, predY)

# tests model on test data, for calculating error uses formula from Kaggle competition
def testModel(model, dataX, dataFile):
    testY = model.predict(dataX)
    saveFile(dataFile, testY)
    return testY

# reads training, cv, randomTesting and testing data for each item, does testing
def doTrainingAndTesting():
    for file in glob.glob(PATH + 'pca13Xcv*train.csv'):
        trainX = readFile(file.replace('Xcv', 'X'))
        trainY = readFile(file.replace('pca13Xcv', 'Y'))
        cvX = readFile(file)
        cvY = readFile(file.replace('pca13X', 'Y'))
        # there is no need for cross-validation for linear regression, because there are no hyper-parameters
        reg = linReg(trainX, trainY)
        regCV = crossValidateModel(reg, cvX, cvY)
        saveFile(file.replace('pca13Xcv', 'regModel'), reg) #saves regression model
        coeffs = np.zeros([6,]) # sets regularization coefficient random samples
        lassoCV = np.zeros([6,])
        ridgeCV = np.zeros([6,])
        svrLinearCV = np.zeros([6, 7])
        svrGaussianCV = np.zeros([6, 7])
        minLassoVal = 10
        minRidgeVal = 10
        minSvmLinear = 10
        minSvmGaussian = 10
        bestLasso = None
        bestRidge = None
        # bestSvmLinear = None
        bestSvmGaussian = None
        for i in range(0, 6):
            coeffs[i] = random.randint(1, 100) / 5 # regularization coefficients
            lasso = lassoReg(trainX, trainY, coeffs[i])
            lassoCV[i] = crossValidateModel(lasso, cvX, cvY)
            if lassoCV[i] < minLassoVal:
                bestLasso = lasso
                minLassoVal = lassoCV[i]
            ridge = ridgeReg(trainX, trainY, coeffs[i])
            ridgeCV[i] = crossValidateModel(ridge, cvX, cvY)
            if ridgeCV[i] < minRidgeVal:
                bestRidge = ridge
                minRidgeVal = ridgeCV[i]
            epsilon =  np.zeros([7, ]) # sets epsilon parameter for SVM
            for j in range(0, 7):
                epsilon[j] = random.randint(1, 1000) / 1000.0
                linearSVR = svrLinear(trainX, trainY, coeffs[i], epsilon[j])
                svrLinearCV[i][j] = crossValidateModel(linearSVR, cvX, cvY)
                if svrLinearCV[i][j] < minSvmLinear:
                    bestSvmLinear = linearSVR
                    minSvmLinear = svrLinearCV[i][j]
                gaussianSVR = svrGaussian(trainX, trainY, coeffs[i], epsilon[j])
                svrGaussianCV[i][j] = crossValidateModel(gaussianSVR, cvX, cvY)
                if svrGaussianCV[i][j] < minSvmGaussian:
                    bestSvmGaussian = gaussianSVR
                    minSvmGaussian = svrGaussianCV[i][j]
        saveFile(file.replace('pca13Xcv', 'lassoModel'), bestLasso)  # saves best lasso model
        saveFile(file.replace('pca13Xcv', 'ridgeModel'), bestRidge) # saves best ridge model
        saveFile(file.replace('pca13Xcv', 'svrLinearModel'), bestSvmLinear) # saves best svmLinear model
        saveFile(file.replace('pca13Xcv', 'svrGaussianModel'), bestSvmGaussian) # saves best svmGaussian model
        bestResults = [regCV, minLassoVal, minRidgeVal, minSvmLinear, minSvmGaussian]
        print(bestResults)
        allResults = [regCV, lassoCV, ridgeCV, svrLinearCV, svrGaussianCV]
        print(allResults)
        saveFile(file.replace('pca13Xcv', 'cvResults'), allResults)
        try:
            testX = readFile(file.replace('Xcv', 'X').replace('train', 'test'))
            testModel(reg, testX, file.replace('pca13Xcv', 'regY').replace('train', 'test'))  # saves regression model predicted values
            testModel(bestLasso, testX, file.replace('pca13Xcv', 'lassoY').replace('train', 'test'))  # saves lasso model predicted values
            testModel(bestRidge, testX, file.replace('pca13Xcv', 'ridgeY').replace('train', 'test'))  # saves ridge model predicted values
            testModel(bestSvmLinear, testX, file.replace('pca13Xcv', 'svrLinearY').replace('train', 'test')) # saves svrLinear model predicted values
            testModel(bestSvmGaussian, testX, file.replace('pca13Xcv', 'svrGaussianY').replace('train', 'test'))  # saves svrGaussian model predicted values
        except FileNotFoundError:
            print('File not found: ' + file.replace('Xcv', 'X').replace('train', 'test'))

def bestMethodsTraining():
    for file in glob.glob('Examples/DeliExample/' + 'pca13Xcv*train.csv'):
        trainX = readFile(file.replace('Xcv', 'X'))
        trainY = readFile(file.replace('pca13Xcv', 'Y'))
        cvX = readFile(file)
        cvY = readFile(file.replace('pca13X', 'Y'))
        gaussianSVR = svrGaussian(trainX, trainY, 800, 0.3)
        gaussianSVR1 = svrGaussian(trainX, trainY, 1000, 0.3)
        svrGaussianCV = crossValidateModel(gaussianSVR, cvX, cvY)
        svrGaussianCV1 = crossValidateModel(gaussianSVR1, cvX, cvY)
        allResults = [svrGaussianCV, svrGaussianCV1]
        print(allResults)

def main():
    # prepareForTraining('train.csv', 125497040, True)
    # prepareForTraining('test.csv', 3370464, False)
    #pca(13)
    #doTrainingAndTesting()
    bestMethodsTraining()

if __name__ == "__main__":
    main()
