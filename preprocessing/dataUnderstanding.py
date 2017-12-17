import csv
import numpy as np
from scipy import stats as s
import matplotlib.pyplot as plt
import datetime
import matplotlib

# reads data from any csv file that is provided and stores it into storeArray
def readData(fileName, rows, cols):
    storeArray = np.empty([rows, cols], dtype=object)
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        i = 0
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                storeArray[i] = np.asarray(row)
                i += 1
    return storeArray

# reads item ids into a dictionary, so that key is item id and value is order of that item in items.csv file
def readItemIds():
    itemIds = {}
    # reads list of all items so that we can use them to group data by items
    with open('items.csv', newline='') as f:
        reader = csv.reader(f)
        i = 0
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                itemIds[row[0]] = i
                i += 1
    return itemIds

# reads and groups data from training by item (row) and week (column); takes data from period 2013-01-01 to 2016-12-31
# weeks are groups of 7 days where each first 7 days of year are 1st week, 8-14 second week and so on
def organizeItemAndWeek():
    itemIds = readItemIds()
    amountPerItemPerWeek = np.zeros([4100, 212], dtype=float)
    # reads in all training data and group them by item and week
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        firstLine = True
        prevday = '2013-01-01'
        year = 0
        i = 0
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                week = int(i / 7)
                # saves data into matrix item x week
                amountPerItemPerWeek[itemIds.get(row[3]), week + 53 * year] += float(row[4])
                if row[1] != prevday:
                    if row[1][0:4] != prevday[0:4]:
                        if (row[1][0:4] == '2017'):
                            break
                        i = 0
                        year += 1
                    else:
                        if row[1][5:] == '12-26':
                            i += 1
                            i += 1
                        i += 1
                    prevday = row[1]
    # writes items x week matrix into a file
    with open('byItemByWeek.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(amountPerItemPerWeek)

# reads and groups data from training by item (row) and week (column); takes data from period 2013-01-01 to 2016-12-31
# weeks are as in calendar, starting from Sunday
def organizeItemAndHumanWeek():
    itemIds = readItemIds()
    amountPerItemPerWeek = np.zeros([4100, 212], dtype=float)
    # reads in all training data and group them by item and week
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        firstLine = True
        prevday = '2013-01-01'
        isodate = datetime.date(2013, 1, 1).isocalendar()
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] != prevday:
                    if (row[1][0:4] == '2017'):
                        break
                    isodate = datetime.date(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10])).isocalendar()
                    prevday = row[1]
                # saves data into matrix item x week
                amountPerItemPerWeek[itemIds.get(row[3]), isodate[1] - 1 + 53 * (isodate[0] - 2013)] += float(row[4])
    # writes items x week matrix into a file
    with open('byItemByHumanWeek.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(amountPerItemPerWeek)

# Wilcoxon and ANOVA tests to discover relationship between items sell in certain week for different years
def distributionsSimilarityTest(fileName):
    data = np.zeros([4100, 212], dtype=float)
    # reads data from item by week sell matrix
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            data[i, :] = np.asfarray(row, float)
            i += 1
    wilcoxonPerWeek = np.zeros([53,], dtype=float)
    anovaPerWeek = np.zeros([53,], dtype=float)
    for i in range(0,53):
            # ANOVA test: compares variation of item sells in same week over years
            anovaPerWeek[i] = s.f_oneway(data[:, i], data[:, i + 53], data[:, i + 106], data[:, i + 159])[1]
            # Wilcoxon signed rank test: compares distributions of item sells in same week over years
            wilcoxonPerWeek[i] = s.wilcoxon(data[:, i], data[:, i + 53])[1] < 0.05 and \
                                 s.wilcoxon(data[:, i + 53], data[:, i + 106])[1] < 0.05 and \
                                 s.wilcoxon(data[:, i + 106], data[:, i + 159])[1] < 0.05 and \
                                 s.wilcoxon(data[:, i], data[:, i + 106])[1] < 0.05 and \
                                 s.wilcoxon(data[:, i], data[:, i + 159])[1] < 0.05 and \
                                 s.wilcoxon(data[:, i + 53], data[:, i + 159])[1] < 0.05
            # since I discovered that for most weeks there is big difference in sells among years,
            # I decided to print and see for which years sale was pretty similar over two years
            # In weeks 27, 28, 31, 35, 49 for years 2015 and 2016 sale distribution was pretty similar
            # Weeks 0 (2014-2015), 23, 29, 42, 44 (2015-2016) has p>0.05, but smaller than 0.19
            if (wilcoxonPerWeek[i] == 0):
                print(str(i) + ' 2013 - 2014 ' + str(s.wilcoxon(data[:, i], data[:, i + 53])[1]) +
                      '; 2014 - 2015 ' + str(s.wilcoxon(data[:, i + 53], data[:, i + 106])[1]) +
                      '; 2015 - 2016 ' + str(s.wilcoxon(data[:, i + 106], data[:, i + 159])[1]) +
                      '; 2013 - 2015 ' + str(s.wilcoxon(data[:, i], data[:, i + 106])[1]) +
                      '; 2013 - 2016 ' + str(s.wilcoxon(data[:, i], data[:, i + 159])[1]) +
                      '; 2014 - 2016 ' + str(s.wilcoxon(data[:, i + 53], data[:, i + 159])[1]))
    print(wilcoxonPerWeek)
    print(sum(wilcoxonPerWeek)/len(wilcoxonPerWeek))
    print(anovaPerWeek)

# visualize relationship between overall sales between years
# visualize overall sales around certain national holidays for each year
# holidays are chosen randomly from national holidays which happen each year on the same date
# graphs have huge variation pattern with length of around 7 days which is repeating
# I am visualizing few random weeks from different parts of year to see if they will have same distribution of sales
def visualizeSalesYearVsYear():
    data = np.zeros([365, 5], dtype=float)
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        firstLine = True
        i = 0
        j = 0
        prevday = '2013-01-01'
        currentYear = '2013'
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if prevday < row[1]:
                    prevday = row[1]
                    i += 1
                if row[1][0:4] > currentYear:
                    currentYear = row[1][0:4]
                    j += 1
                    i = 0
                data[i, j] += float(row[4])
    # normalizes values by maximum in each list, then plots list for different years
    for i in range(0, 5):
        data[:, i] /= np.max(data[:, i])
        plt.plot(data[:, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('salesYearVsYear.png', dpi=900)
    plt.clf()
    # to understand better changes around holidays or within the week, plot just appropriate parts of the year
    # I took bigger range around holidays to understand holiday effects
    # 4-5 days around the holiday is enough to see its effect and relation with the week part in which it is happening
    for i in range(0, 5):
        plt.plot(data[111:132, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Labor Day 1st May.png', dpi=900)
    plt.clf()
    for i in range(0, 5):
        plt.plot(data[134:155, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Battle of Pichincha 24th May.png', dpi=900)
    plt.clf()
    for i in range(0, 5):
        plt.plot(data[212:232, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Ecuadorian War of Independence 10th August.png', dpi=900)
    plt.clf()
    for i in range(0, 5):
        plt.plot(data[297:318, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Independence of Cuenca 3rd November.png', dpi=900)
    plt.clf()
    for i in range(0, 5):
        plt.plot(data[245:252, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Week 35.png', dpi=900)
    plt.clf()
    for i in range(0, 5):
        plt.plot(data[63:70, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Week 9.png', dpi=900)
    plt.clf()
    for i in range(0, 5):
        plt.plot(data[329:336, i])
    plt.legend(['2013', '2014', '2015', '2016', '2017'])
    plt.savefig('Week 47.png', dpi=900)
    plt.clf()

# visualize relationship between overall sales amount and oil price
def visualizeSalesVsOilPrice():
    oilVsSales = np.zeros([1218, 3], dtype=object)
    oilVsSales[:, 0:2] = readData('oil.csv', 1218, 2)
    i = 0
    while i < len(oilVsSales):
        if (oilVsSales[i, 1] == ''):
            oilVsSales = np.delete(oilVsSales, i, 0)
        else:
            i += 1
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        firstLine = True
        i = 0
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] > oilVsSales[i, 0]:
                    i+=1
                elif row[1] == oilVsSales[i, 0]:
                    oilVsSales[i, 2] += float(row[4])
    # normalize values by maximum in each list, to easier compare trends
    oilTrend = np.asfarray(oilVsSales[:, 1], float) / np.max(np.asfarray(oilVsSales[:, 1], float))
    salesTrend = np.asfarray(oilVsSales[:, 2], float) / np.max(np.asfarray(oilVsSales[:, 2], float))
    plt.plot(oilVsSales[:, 0], oilTrend)
    plt.plot(oilVsSales[:, 0], salesTrend)
    plt.legend(['oil trend', 'sales trend'])
    plt.savefig('oilVsSales.png')
    plt.clf()

# gets info about cities, states, types and clusters of stores
def getStoresInfo():
    citiesSales = {}
    storesPerCity = {}
    statesSales = {}
    storesPerState = {}
    storeTypeSales = {}
    storesPerType = {}
    storeClusterSales = {}
    storesPerCluster = {}
    stores = np.empty([54, 5], dtype=object)
    with open('stores.csv', newline='') as f:
        reader = csv.reader(f)
        i = 0
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] not in citiesSales:
                    citiesSales[row[1]] = 0.0
                    storesPerCity[row[1]] = 1
                else:
                    storesPerCity[row[1]] += 1
                if row[2] not in statesSales:
                    statesSales[row[2]] = 0.0
                    storesPerState[row[2]] = 1
                else:
                    storesPerState[row[2]] += 1
                if row[3] not in storeTypeSales:
                    storeTypeSales[row[3]] = 0.0
                    storesPerType[row[3]] = 1
                else:
                    storesPerType[row[3]] += 1
                if row[4] not in storeClusterSales:
                    storeClusterSales[row[4]] = 0.0
                    storesPerCluster[row[4]] = 1
                else:
                    storesPerCluster[row[4]] += 1
                stores[i, :] = np.asarray(row)
                i += 1
    return [citiesSales, storesPerCity, statesSales, storesPerState, storeTypeSales, storesPerType, storeClusterSales, storesPerCluster, stores]

# draws bar plot from dictionary and saves it into file with filename
def drawBarPlot(dictionary, filename):
    plt.bar(range(len(dictionary)), dictionary.values())
    plt.xticks(range(len(dictionary)), dictionary.keys())
    plt.savefig(filename, dpi=900)
    plt.clf()

# visualise sales differences between different store types/clusters/cities/states
def visualizeSalesVsStoreTypeOrCluster():
    citiesSales, storesPerCity, statesSales, storesPerState, storeTypeSales, storesPerType, storeClusterSales, storesPerCluster, stores = getStoresInfo()
    # reads list of all items so that we can use them to group data by family or classes
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] >= '2013-02-03':
                    break
                # aggregates sales amount per city, state, storeType, storeCluster
                store = stores[int(row[2]) - 1]
                citiesSales[store[1]] += float(row[4])
                statesSales[store[2]] += float(row[4])
                storeTypeSales[store[3]] += float(row[4])
                storeClusterSales[store[4]] += float(row[4])
    # normalizes values by maximum in each dictionary, to be able to compare trends
    maxCities = max(citiesSales.values())
    maxStates = max(statesSales.values())
    maxStoreType = max(storeTypeSales.values())
    maxCluster = max(storeClusterSales.values())
    citiesSales = {k: v / maxCities / storesPerCity[k]  for k, v in citiesSales.items()}
    statesSales = {k: v / maxStates / storesPerState[k] for k, v in statesSales.items()}
    storeTypeSales = {k: v / maxStoreType / storesPerType[k] for k, v in storeTypeSales.items()}
    storeClusterSales = {k: v / maxCluster / storesPerCluster[k] for k, v in storeClusterSales.items()}
    drawBarPlot(citiesSales, 'citiesSales.png')
    drawBarPlot(statesSales, 'statesSales.png')
    drawBarPlot(storeTypeSales, 'storeTypeSales.png')
    drawBarPlot(storeClusterSales, 'storeClusterSales.png')

# gets info about classes and families of items
def getItemsInfo():
    itemFamily = {}
    itemClass = {}
    itemFamilyNoSold = {}
    itemClassNoSold = {}
    itemsPerFamily = {}
    itemsPerClass = {}
    items = np.empty([4100, 4], dtype=object)
    # reads list of all items so that we can use them to group data by family or classes
    with open('items.csv', newline='') as f:
        reader = csv.reader(f)
        i = 0
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] not in itemFamily:
                    itemFamily[row[1]] = 0.0
                    itemFamilyNoSold[row[1]] = 0
                    itemsPerFamily[row[1]] = 1
                else:
                    itemsPerFamily[row[1]] += 1
                if row[2] not in itemClass:
                    itemClass[row[2]] = 0.0
                    itemClassNoSold[row[2]] = 0
                    itemsPerClass[row[2]] = 1
                else:
                    itemsPerClass[row[2]] += 1
                items[i, :] = np.asarray(row)
                i += 1
    return [itemFamily, itemClass, itemFamilyNoSold, itemClassNoSold, itemsPerFamily, itemsPerClass, items]

# visualise sales differences between different items and item categories
# amount of items bought shouldn't be taken into consideration because of different packages and sizes (kg, g, l, ml)
# good to see which types of items are not sold at all and what are best selling categories for this chain
# variables ending with NoSold contain information about popularity of itemFamilies/itemClasses;
def visualizeSalesVsItemFamilyOrClass():
    itemIds = readItemIds()
    itemFamily, itemClass, itemFamilyNoSold, itemClassNoSold, itemsPerFamily, itemsPerClass, items = getItemsInfo()
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        firstLine = True
        for row in reader:
            if firstLine:
                firstLine = False
            else:
                if row[1] >= '2013-03-03':
                    break
                # aggregate items amount per item family and class
                item = items[itemIds.get(row[3])]
                itemFamilyNoSold[item[1]] += 1
                itemClassNoSold[item[2]] += 1
                itemFamily[item[1]] += float(row[4])
                itemClass[item[2]] += float(row[4])
                # normalize values by maximum in each dictionary, to be able to compare trends
    maxFamily = max(itemFamily.values())
    maxClass = max(itemClass.values())
    itemFamily = {k: v / maxFamily / itemsPerFamily[k] for k, v in itemFamily.items()}
    itemClass = {k: v / maxClass / itemsPerClass[k] for k, v in itemClass.items()}
    itemFamilyNoSold = {k: v / itemsPerFamily[k] for k, v in itemFamilyNoSold.items()}
    itemClassNoSold = {k: v / itemsPerClass[k] for k, v in itemClassNoSold.items()}
    drawBarPlot(itemFamily, 'itemFamily.png')
    drawBarPlot(itemClass, 'itemClass.png')
    drawBarPlot(itemFamilyNoSold, 'itemFamilyNoSold.png')
    drawBarPlot(itemClassNoSold, 'itemClassNoSold.png')

def main():
    # Please, change font size as it fits you. I was comparing some graphs with many classes and long names
    # resolution is high so it can be read even with small font; normal size is 12
    matplotlib.rcParams.update({'font.size': 1})
    organizeItemAndWeek()
    organizeItemAndHumanWeek()
    distributionsSimilarityTest('byItemByWeek.csv')
    distributionsSimilarityTest('byItemByHumanWeek.csv')
    visualizeSalesYearVsYear()
    visualizeSalesVsOilPrice()
    visualizeSalesVsStoreTypeOrCluster()
    visualizeSalesVsItemFamilyOrClass()

if __name__ == "__main__":
    main()
