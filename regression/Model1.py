from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
import time
import pickle



def handling_null_values_with_median(dataframe):
    dataframe=dataframe.fillna(dataframe.median())

    return dataframe

def handling_null_values_with_mean(dataframe):
    dataframe=dataframe.fillna(dataframe.mean())

    return dataframe

def handling_null_values_with_mode(dataframe):
    dataframe=dataframe.fillna(dataframe.mode().iloc[0])

    return dataframe

def to_null(dataframe,value):
    print(len(dataframe))
    for i in range(len(dataframe)):
        for j in value:
            if dataframe[i] == j:
                dataframe[i] = np.nan
    return dataframe

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

def one_hot_encoder (df , data):
    label_encoder = LabelEncoder()
    # Encode labels in column 'Country'.
    data['LoanStatus'] = label_encoder.fit_transform(data['LoanStatus'])
    # one hot encoder

    onehotencoder = OneHotEncoder()
    x = onehotencoder.fit_transform(data.LoanStatus.values.reshape(-1, 1)).toarray()

    dfOneHot = pd.DataFrame(x, columns=["LoanStatus_" + str(int(i)) for i in range(12)])
    df = pd.concat([data, dfOneHot], axis=1)
    df = df.drop(['LoanStatus'], axis=1)

    df.drop(['LoanStatus_11'], axis=1, inplace=True)

    return df , data


def process_on_homeowner (X):
     lbl = LabelEncoder()
     lbl.fit(list(X['IsBorrowerHomeowner'].values))
     X['IsBorrowerHomeowner'] = lbl.transform(list(X['IsBorrowerHomeowner'].values))
     return X


def process_on_incomerange_withdoutdisplayed (df):
    for i in range (len(df)):

        if df.iat[i , 18] == 'Not employed':
            df.iat[i , 18] = 0

        if df.iat[i, 18] == '$0 ':
            df.iat[i, 18] = 1

        if df.iat[i, 18] == '$1-24,999':
            df.iat[i, 18] = 2

        if df.iat[i, 18] == '$25,000-49,999':
            df.iat[i, 18] = 3

        if df.iat[i, 18] == '$50,000-74,999':
            df.iat[i, 18] = 4

        if df.iat[i, 18] == '$75,000-99,999':
            df.iat[i, 18] = 5

        if df.iat[i, 18] == '$100,000+':
            df.iat[i, 18] = 6




        return df


def process_on_incomerange_withdisplayed(df):
    for i in range(len(df)):

        if df.iat[i, 18] == 'Not employed':
            df.iat[i, 18] = 0

        if df.iat[i, 18] == '$0 ':
            df.iat[i, 18] = 1

        if df.iat[i, 18] == '$1-24,999':
            df.iat[i, 18] = 2

        if df.iat[i, 18] == '$25,000-49,999':
            df.iat[i, 18] = 3

        if df.iat[i, 18] == '$50,000-74,999':
            df.iat[i, 18] = 4

        if df.iat[i, 18] == '$75,000-99,999':
            df.iat[i, 18] = 5

        if df.iat[i, 18] == '$100,000+':
            df.iat[i, 18] = 6

        if df.iat[i, 18] == 'Not displayed':
            df.iat[i, 18] = 7

    return df


def preprocessing(dataframe):
    dataframe.drop(['CreditGrade'], axis=1, inplace=True)  # axis = 1 column axis = 0 rows
    dataframe.drop(['TotalProsperPaymentsBilled'], axis=1, inplace=True)

    Number_cols = ["BorrowerAPR", "EmploymentStatusDuration", "CreditScoreRangeLower", "CreditScoreRangeUpper",
                   "RevolvingCreditBalance", "BankcardUtilization", "AvailableBankcardCredit", "TotalTrades",
                   "DebtToIncomeRatio"]
    dataframe[Number_cols] = dataframe[Number_cols].fillna(dataframe.median())

    String_cols = ["BorrowerState", "EmploymentStatus"]
    dataframe[String_cols] = dataframe[String_cols].fillna(dataframe.mode().iloc[0])

    dataframe['LoanRiskScore'].fillna(dataframe['LoanRiskScore'].mean(), inplace=True)
     # ely homa strings h3mlhom encoding


    return dataframe
def model_preprocessing(X):
    cols = ['LoanStatus', 'BorrowerState', 'EmploymentStatus', 'IsBorrowerHomeowner',
            'IncomeRange']
    X = Feature_Encoder(X, cols)

    return X


data = pd.read_csv('LoanRiskScore.csv')
df = pd.DataFrame(data)
df=preprocessing(df)
 # ely homa strings h3mlhom encoding
X = df.iloc[:, 1:21]  # Features
Y = df['LoanRiskScore']  # Label

X=model_preprocessing(X)


#X = ps.featureScaling(X, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2)


# determine the mutual information

mutual_info = mutual_info_regression(X_train, y_train)
mutual_info = pd.Series(mutual_info)
# mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)
mutual_info.sort_values(ascending=False).plot.bar(figsize=(100, 100))


#rfr = RandomForestRegressor(n_estimators=50, max_depth=10)
rfr = RandomForestRegressor(n_estimators=60, max_depth=16,random_state=1111)
start = time.time()
rfr.fit(X_train, y_train)
stop = time.time()
testpredictions = rfr.predict(X_test)
meansquareerror = metrics.mean_squared_error(y_test, testpredictions)
print('Accuracy ',metrics.r2_score(y_test,testpredictions))
print(f"Training time: {stop - start}s")
print('MSE',meansquareerror)
# Split the data to training and testing sets

Pkl_Filename = "RFR 1_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
   pickle.dump(rfr, file)

'''with open(Pkl_Filename, 'rb') as file:
    Pickled_LR_Model = pickle.load(file)

# Calculate the Score
score = Pickled_LR_Model.score(X_test, y_test)
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(X_test)
meansquareerror = metrics.mean_squared_error(y_test, Ypredict)
print("Y Predicted ", Ypredict)
print("MSE ", meansquareerror)'''
