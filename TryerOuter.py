from csvLoader import csv_loader
import sys
import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import metrics

def main():
    contents = csv_loader('FoodNutrients.csv')
    # print(contents)
    nullsByColumn = list()
    floatableObjByColumn = list()
    ColumnNames = contents[0]
    nameShouldBeKept = list()
    columnHeadsmansList = set()
    shouldBeKept = [3,6,8,10,12,13,14,15,16,17,18,19,22,23,24,25,34]
    for i in range(len(shouldBeKept)):
        nameShouldBeKept.insert(0,ColumnNames[shouldBeKept[i]])
    for i in range(len(contents[0])):
        if i in shouldBeKept:
            pass
        else:
            columnHeadsmansList.add(i)
    print("--------------------------------------------------")
    print("Data Dimensions")
    print('Columns: '+str(len(contents[0])))
    print('Rows: '+str(len(contents)))
    contents = purgeColumn(contents, columnHeadsmansList)
    # print(contents)
    dataframeList = list()
    rowHeadsmansList = set()
    rowHeadsmansList.add(0)
    rowHeadsmansList.add(1)
    nullAmt = 0
    for i in range(len(contents)):
        for j in range(len(contents[0])):
            if contents[i][j] == '':
                nullAmt+=1
    print("--------------------------------------------------")
    print("Amount of Null Values")
    print(nullAmt)
    contents = purgeRows(contents,rowHeadsmansList)
    for i in range(len(contents[0])):
        columnList = list()
        for j in range(len(contents)):
            columnList.append(contents[j][i])
        # print(columnList)
        cdf = pd.DataFrame(columnList, dtype='O')
        cdf = cdf.apply(pd.to_numeric, errors='coerce', downcast='float')
        dataframeList.append(cdf)
    # print("--------------------------------------------------")
    # print("Pre-Interpolation Dataframe")
    # print(dataframeList)
    for i in range(len(dataframeList)):
        dataframeList[i] = dataframeList[i][0].interpolate(method='linear',limit_direction='both')
    # print("--------------------------------------------------")
    # print("Post-Interpolation Dataframe")
    # print(dataframeList)

    for i in range(len(dataframeList)):
        dataframeList[i] = dataframeList[i].values.tolist()
    print(dataframeList)

    derivedColumn = dataframeList[0]

    # print("--------------------------------------------------")
    # print("Class Values")
    # print(derivedColumn)

    print("--------------------------------------------------")
    print('DataFrameList')
    print(dataframeList)

    featuresColumn = dataframeList[1:]
    derivedColumnName = ColumnNames[0]
    featureColumnNames = ColumnNames[1:]
    featuresColumn = np.array(featuresColumn).T.tolist()
    # featuresColumn = StandardScaler().fit_transform(featuresColumn)

    x_train, x_test, y_train, y_test = train_test_split(featuresColumn, derivedColumn, test_size=0.66, random_state=1, shuffle=True)

    # print("--------------------------------------------------")
    # print("FeatureColumn Train")
    # print(x_train)
    # print("--------------------------------------------------")
    # print("FeatureColumn Test")
    # print(x_test)
    # print("--------------------------------------------------")
    # print("FeatureColumn Train")
    # print(y_train)
    # print("--------------------------------------------------")
    # print("FeatureColumn Test")
    # print(y_test)

    # print("--------------------------------------------------")
    # print("Features Values")
    # print(featuresColumn)

    trainLength = (len(derivedColumn)*2)//3
    print("Total Length")
    print(len(derivedColumn))
    print("Train Length")
    print(trainLength)
    # for i in length

    GaussianNBPCAModel = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    GaussianNBPCAModel.fit(x_train, y_train)
    GaussianNBPCAModelTest = GaussianNBPCAModel.predict(x_test)

    print('\nPerformance of Gaussian NB')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, GaussianNBPCAModelTest)))


    x_train, x_test, y_train, y_test = train_test_split(featuresColumn, derivedColumn, test_size=0.66, random_state=1, shuffle=True)

    for i in range(1,6):
        polyreg = make_pipeline(StandardScaler(), PCA(n_components=8), PolynomialFeatures(i), LinearRegression())
        polyreg.fit(x_train, y_train)
        val_acc = polyreg.score(x_test, y_test)
        print("Predicted Values")
        predval = np.array(polyreg.predict(x_test))
        print(np.round(predval))
        print('\nPerformance of Polynomial Regression on degree '+str(i))
        print('{:.2%}\n'.format(val_acc))

    x_train, x_test, y_train, y_test = train_test_split(featuresColumn, derivedColumn, test_size=0.66, random_state=1, shuffle=True)

    clf = linear_model.Lasso(alpha=1)
    clf.fit(x_train, y_train)
    print("Coefficient:", clf.coef_)
    print("Intercept:", clf.intercept_)
    acc = clf.score(x_test, y_test)
    print("Predicted Values")
    print(clf.predict(x_test))
    print('\nPerformance of Lasso')
    print('{:.2%}\n'.format(acc))

    x_train, x_test, y_train, y_test = train_test_split(featuresColumn, derivedColumn, test_size=0.66, random_state=1, shuffle=True)
    setup = make_pipeline(StandardScaler(), PCA(n_components=8))
    x_train, x_test, y_train, y_test = setup.fit(x_train, x_test, y_train, y_test)
    regr = MLPRegressor(random_state=0, max_iter=500).fit(x_train, y_train)
    print(regr.predict(x_test[:2]))

    sys.exit()

    n = int(abs(len(cleansedList)*0.66))
    print("Training Data Size")
    print(n)
    classList = list() #label
    featuresList = list() #variables
    fullList = list()
    for i in range(len(cleansedList)):
        classList.append(str(cleansedList[i][classColumn])[0:2])
        frontTuple = cleansedList[i][:classColumn]
        backTuple = cleansedList[i][classColumn+1:] #REMOVE IF DATA IS AT EITHER END OF THE COLUMN
        featuresList.append(frontTuple+backTuple)
    finalFeatures = finalPurge(featuresList,3) #we want only 3 features
    finalFeaturesNames = []
    finalPurgeIndexes = []
    for i in range(len(ColumnNames)):
        if i not in finalFeatures:
            finalPurgeIndexes.append(i)
        else:
            finalFeaturesNames.append(ColumnNames[i])
    print("Final Features Names")
    print(finalFeaturesNames)
    featuresList = purgeColumn(featuresList,finalPurgeIndexes)
    # print(featuresList)
    classifiers = set()
    for i in range(len(featuresList)):
        fullTuple = list()
        for j in range(len(featuresList[0])):
            fullTuple.append(featuresList[i][j])
        if classList[i]==1.0:
            fullTuple.append('FEMALE')
        elif classList[i]==0.0:
            fullTuple.append('MALE')
        else:
            fullTuple.append("a"+str(classList[i]))
        fullList.append(fullTuple)
    print("Final Features: "+str(finalFeatures))
    trainingClass = list()
    trainingFeatures = list()
    validationClass = list()
    validationFeatures = list()
    featuresList = StandardScaler().fit_transform(featuresList)
    # classList = StandardScaler().fit_transform(classList)
    trainingClass = classList[:n]
    validationClass = classList[n:]
    trainingFeatures = featuresList[:n]
    validationFeatures = featuresList[n:]
    maleCount, femaleCount, trainMaleCount, trainFemaleCount = 0,0,0,0
    for i in trainingClass:
        if i == 0:
            trainMaleCount+=1
        else:
            trainFemaleCount+=1
    for i in validationClass:
        if i == 0:
            maleCount+=1
        else:
            femaleCount+=1


    print("Validation Features: " + str(len(validationFeatures)))
    print("Validation Class: " + str(len(validationClass)))
    print("Training Features: " + str(len(trainingFeatures)))
    print("Training Class: " + str(len(trainingClass)))
    # print(validationClass)
    # print(trainingClass)
    print("Training Data Male Count: " + str(trainMaleCount))
    print("Training Data Female Count: " + str(trainFemaleCount))
    print("Validation Data Male Count: " + str(maleCount))
    print("Validation Data Female Count: " + str(femaleCount))

    with open('trainingFeatures.csv', mode='w', newline='') as trainingFeaturesFile:
        trainingFeaturesWriter = csv.writer(trainingFeaturesFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(trainingFeatures)):
            trainingFeaturesWriter.writerow(trainingFeatures[i])
    with open('trainingClass.csv', mode='w', newline='') as trainingClassFile:
        trainingClassWriter = csv.writer(trainingClassFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trainingClassWriter.writerow(trainingClass)
    with open('validationFeatures.csv', mode='w', newline='') as validationFeaturesFile:
        validationFeaturesWriter = csv.writer(validationFeaturesFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(validationFeatures)):
            validationFeaturesWriter.writerow(validationFeatures[i])
    with open('validationClass.csv', mode='w', newline='') as validationClassFile:
        validationClassWriter = csv.writer(validationClassFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        validationClassWriter.writerow(validationClass)
    with open('ClassAndRace.csv', mode='w', newline='') as classAndRaceFile:
        fullWriter = csv.writer(classAndRaceFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(fullList)):
            fullWriter.writerow(fullList[i])
    # print(trainingFeatures)


    # for i in range(len(cleansedList[0])):

#fifthColumnists are columns with too much nulls or no floatable objects
def purgeColumn(contents, fifthColumnists):
    listOfTuples = list()
    for i in range(len(contents)):
        thisTuple = list()
        for j in range(len(contents[0])):
            if j in fifthColumnists:
                pass
            else:
                thisTuple.append(contents[i][j])
        tuple(thisTuple)
        listOfTuples.append(thisTuple)
    # #do this for mean
    # #alternatively, drop the row.


    # meanList = list()
    # # print(listOfTuples)
    # for i in range(len(listOfTuples[0])):
    #     thisColumnMean = 0
    #     zeroCounter = 0
    #     # print(listOfTuples[j])
    #     for j in range(len(listOfTuples)):
    #         try:
    #             thisColumnMean += listOfTuples[j][i]
    #         except TypeError:
    #             zeroCounter +=1
    #             thisColumnMean += 0
    #     if zeroCounter>=1:
    #         thisColumnMean = thisColumnMean//(len(listOfTuples)-zeroCounter)
    #     else:
    #         pass
    #     meanList.append(thisColumnMean)
    # for i in range(len(listOfTuples)):
    #     for j in range(len(listOfTuples[0])):
    #         if listOfTuples[i][j] == '':
    #             listOfTuples[i][j] = meanList[j]


    # print(meanList)
    return listOfTuples

#reactionaries are dissident rows that has null values in it
def purgeRows(contents, reactionaries):
    listOfTuples = list()
    for i in range(len(contents)):
        if i != 0:
            if i in reactionaries:
                pass
            else:
                thisTuple = list()
                for j in range(len(contents[0])):
                    thisTuple.append(contents[i][j])
                listOfTuples.append(thisTuple)
    return listOfTuples

def meanNer(column):
    sum = sum(column)
    mean = sum//len(column)
    return mean



def finalPurge(contents,desiredDim):
    # print(contents)
    features = set()
    # print(contents)
    #standardize the entire contents
    contents = StandardScaler().fit_transform(contents)
    npArrayContents = np.array(contents)
    npArrayContents = npArrayContents.T
    print(npArrayContents.tolist())
    cov = np.cov(npArrayContents,bias=True)
    eigenvalues,eigenvectors = np.linalg.eig(cov)
    print("Length of eigenvalues: "+ str(len(eigenvalues)))
    print("Length of eigenvectors: "+ str(len(eigenvectors)))
    eigenPairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i],i) for i in range(len(eigenvalues))]
    eigenPairs.sort(key=takeKey,reverse=True)
    for i in range(desiredDim):
        features.add(eigenPairs[i][2])
    print("Length of EigenPairs: "+ str(len(eigenPairs)))
    return features
    #find eigenvalues of each column
    #find column I want to classify
    #return top 4 eigenvalue and classifier column

def takeKey(eigPair):
    return eigPair[0]

if __name__ == "__main__":
    main()
