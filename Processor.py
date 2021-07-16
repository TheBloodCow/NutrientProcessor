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
    #Columns that should be kept here is arbitrarily picked from the dataset, column 3 in this case is the selected feature
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

    #Both transposes the data (to a column-oriented form) and casts float on all data points, incompatible data is NaN'd and replaced later

    for i in range(len(dataframeList)):
        dataframeList[i] = dataframeList[i][0].interpolate(method='linear',limit_direction='both')

    # Linear interpolation with both limit_direction to fill in NaN's to both ends of the data

    # print("--------------------------------------------------")
    # print("Post-Interpolation Dataframe")
    # print(dataframeList)

    for i in range(len(dataframeList)):
        dataframeList[i] = dataframeList[i].values.tolist()
    # print(dataframeList)

    derivedColumn = dataframeList[0]

    # print("--------------------------------------------------")
    # print("Class Values")
    # print(derivedColumn)

    # print("--------------------------------------------------")
    # print('DataFrameList')
    # print(dataframeList)

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

    # print("Total Data Length")
    # print(len(derivedColumn))
    # print("Training Data Length")
    # print(trainLength)

    GaussianNBPCAModel = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    GaussianNBPCAModel.fit(x_train, y_train)
    GaussianNBPCAModelTest = GaussianNBPCAModel.predict(x_test)

    print('\nPerformance of Gaussian NB')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, GaussianNBPCAModelTest)))

    # Apparently the dataset doesn't form a Gaussian Distribution in the slightest.

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

    # Polynomial Regression with higher degrees does not create a workable model (above 2)
    # Polynomial of degree 1 is linear Regression
    # Negative percentage Result on higher degrees

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

    # Lasso is highly accurate in this case and is best suited for it.

    x_train, x_test, y_train, y_test = train_test_split(featuresColumn, derivedColumn, test_size=0.66, random_state=1, shuffle=True)
    setup = make_pipeline(StandardScaler(), PCA(n_components=8))
    x_train, x_test, y_train, y_test = setup.fit(x_train, x_test, y_train, y_test)
    regr = MLPRegressor(random_state=0, max_iter=500).fit(x_train, y_train)
    print(regr.predict(x_test[:2]))

    sys.exit()

# undesirableColumns are, in this case, those that contain string values or unfloatable values
# Idea: simplify by using pandas and transposing it.
def purgeColumn(contents, undesirableColumns):
    listOfTuples = list()
    for i in range(len(contents)):
        thisTuple = list()
        for j in range(len(contents[0])):
            if j in undesirableColumns:
                pass
            else:
                thisTuple.append(contents[i][j])
        tuple(thisTuple)
        listOfTuples.append(thisTuple)
    return listOfTuples

# undesirableRows are rows that should be removed, in this case just the topmost label and their units of measurement.
def purgeRows(contents, undesirableRows):
    listOfTuples = list()
    for i in range(len(contents)):
        if i != 0:
            if i in undesirableRows:
                pass
            else:
                thisTuple = list()
                for j in range(len(contents[0])):
                    thisTuple.append(contents[i][j])
                listOfTuples.append(thisTuple)
    return listOfTuples

if __name__ == "__main__":
    main()
