import pandas as pd
import numpy as np

#   MISSING ----------------------------------------------------------------
def showNull(data): #Print missing values
    null_values = data.isnull().sum()
    index = 0
    for i in null_values:
        if i > 0:
            print(data.columns[index], "   ", i)
        index += 1
    print("\n", "(Columns: ", str(index)+")")

def showMean(data): #Print the mean of numerical columns
    numerical = data.select_dtypes(include=np.number)
    for column in numerical:
        serie = data[column].mean()
        print(column, serie)

def showMedian(data): #Print the mean of numerical columns
    numerical = data.select_dtypes(include=np.number)
    for column in numerical:
        serie = data[column].median()
        print(column, serie)

def showMode(data): #Print info about the mode of categorical data
    categorical = data.select_dtypes(exclude=np.number)
    for column in categorical:
        ex = data[column]
        print(ex.value_counts())
        print("")
        print(ex.mode())
        print("")
        print(ex.mode().values)
        print("\n")

def fillAsMean(data, column): #Fill missing values of a column with the mean
    serie = data[column].mean()
    data[column].fillna(value=serie, inplace=True)

def numericalAsMean(data, include=[], exclude=[]): #Fill the missing values of numerical data with the mean
    if include:
        numerical = include
    else:
        numerical = data.select_dtypes(include=np.number)
    for column in numerical:
        if column not in exclude:
            fillAsMean(data, column)

def fillAsMedian(data, column): #Fill m. values of a  column with the median
    serie = data[column].median()
    data[column].fillna(value=serie, inplace=True)

def numericalAsMedian(data, include=[], exclude=[]): #Fill the m. values of numerical data with median
    if include:
        numerical = include
    else:
        numerical = data.select_dtypes(include=np.number)
    for column in numerical:
        if column not in exclude:
            fillAsMedian(data, column)

def fillAsMode(data, column): #Fill m. values of a column with the mode
    array = data[column].mode().values
    serie = array[0]
    data[column].fillna(value=serie, inplace=True)

def categoricalAsMode(data): #Fill m. values of categorical data with the mode
    categorical = data.select_dtypes(exclude=np.number)
    for column in categorical:
        array = data[column].mode().values
        serie = array[0]
        data[column].fillna(value=serie, inplace=True)




#   MAPPING ----------------------------------------------------------------
def showColumns(data):
    print(data.columns)

def showCategories(data, feature):
    count = data[feature].value_counts()
    index = count.index
    array = index.values
    categories = np.ndarray.tolist(array)
    print(type(count))
    print(count)
    print()
    print(type(index))
    print(index)
    print()
    print(type(array))
    print(array)
    print()
    print(type(categories))
    print(categories)

def getCategories(data, feature):
    array = data[feature].value_counts().index.values
    categories = np.ndarray.tolist(array)
    return categories

def mapCategories(data, feature): #return a dictionary the the categories of a single feature
    array = data[feature].value_counts().index.values #return the categories in a numpy array
    categories = np.ndarray.tolist(array) #convert that array in a native python list
    book = {i:categories[i] for i in range(0, len(categories))} #ennumerate the categories
    myDict = {i: j for j, i in book.items()} #invert the keys and values of the prev. dict
    return myDict

def categoricalMapping(data, include=[], exclude=[]): #do the same for all categorical features
    if include:
        categorical = include
    else:
        categorical = data.select_dtypes(exclude=np.number)
    for feature in categorical:
        if feature not in exclude:
            mapCategories(data, feature)

def writeCategories(data, feature, book={}): #overwrite the features converted into numbers
    if not book:
        book = mapCategories(data, feature)
    else:
        book = book
    data[feature] = data[feature].map(book)

def categoricalWritting(data, include=[], exclude=[]): #do the same for all the dataset
    if include:
        categorical = include
    else:
        categorical = data.select_dtypes(exclude=np.number)
    for feature in categorical:
        if feature not in exclude:
            writeCategories(data, feature)




#   TRY THE MODELS -----------------------------------------------------------
import tensorflow as tf
from sklearn.model_selection import KFold as KFold
from sklearn.model_selection import cross_val_score as VScore
from sklearn.metrics import mean_squared_error, r2_score

class myCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        
    def on_epoch_end(self, epoch, logs):
        if(logs.get("accuracy") > 0.999):
            self.model.stop_training= True
            print("\nReached 99.9% of accuracy so cancelling training")


def getAccuracy(model, data, target, parameters=""):
    model = eval(model)()
    if parameters:
        string = "model.set_params("+parameters+")"
        exec(string)
    scoring = ("neg_mean_squared_error", "r2")
    folds = np.abs(VScore(model, data, target, cv=10, n_jobs=1, scoring=scoring[0]))
    mean_score = np.mean(folds)
    r2_score = np.mean(VScore(model, data, target, cv=10, n_jobs=1, scoring=scoring[1]))
    return folds, mean_score, r2_score

def tryModel(model, data, target, parameters=""):
    folds, mean_score, r2_score = getAccuracy(model, data, target, parameters)
    print("KFold Scores:  ", folds, "\n")
    print("MEAN SQUARED ERROR:  ", format(mean_score, ".4e"), "\n")
    print("R2 ACCURACY:  ", "{0:.5g}".format(r2_score*100), "%")

def testModels(data, target, models):
    folds = [] #kfolds
    means = [] #squared error
    scores = [] #r2 score
    #Compute each model and print the progress
    unit = 100/len(models)
    progress = unit
    for i in models:
        try:
            print("Running", i + "...")
            folds.append(getAccuracy(i, data, target)[0])
            means.append(getAccuracy(i, data, target)[1])
            scores.append(getAccuracy(i, data, target)[2])
            print(i, "model successfully computed. Progress: ", "{0:.3g}".format(progress), "%")
            progress += unit
        except:
            print("ERROR!!! The model ", i, "was NOT computed")
    print("="*64, "\n")

    #Decide what model is the best and and printed on screen
    max_value = max(scores)
    index = scores.index(max_value)
    k_folds = folds[index]
    mean_score = means[index]
    print("BEST MODEL:  ", models[index], "\n")
    print("KFold Scores:  ", k_folds, "\n")
    print("MEAN SQUARED ERROR:  ", format(mean_score, ".4e"), "\n")
    print("R2 ACCURACY: ",  "{0:.5g}".format(max_value*100), "%")
