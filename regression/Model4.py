import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import statistics
import warnings
import missingno as msno
from sklearn.feature_selection import mutual_info_regression
import time

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
data = pd.read_csv('LoanRiskScore.csv')
olddf = pd.DataFrame(data)

olddf  = process_on_homeowner(olddf)
olddf = process_on_incomerange_withdisplayed(olddf)
olddf = one_hot_encoder(olddf , data)

df = olddf[0]

df.drop(['ListingNumber'],axis = 1 ,inplace=True)
df.drop(['CreditGrade'],axis = 1 ,inplace=True)#axis = 1 column axis = 0 rows
df.drop(['TotalProsperPaymentsBilled'],axis = 1 ,inplace=True)
print(df.isna().sum())
print(len(df))


Number_cols=['BorrowerAPR','EmploymentStatusDuration','CreditScoreRangeLower','CreditScoreRangeUpper','RevolvingCreditBalance','BankcardUtilization','AvailableBankcardCredit','TotalTrades','DebtToIncomeRatio']
numerical_features=df[Number_cols]
df[Number_cols]=handling_null_values_with_median(numerical_features)

String_cols=['BorrowerState','EmploymentStatus']
categorical_features=df[String_cols]
df[String_cols]=handling_null_values_with_mode(categorical_features)
df['LoanRiskScore']=handling_null_values_with_mean(df['LoanRiskScore'])
print(df.isna().sum())

X = df.iloc[:, df.columns != 'LoanRiskStatus']  # Features
Y = df['LoanRiskScore']  # Label
cols = ['BorrowerState','EmploymentStatus']
X = Feature_Encoder(X, cols)


#X = ps.featureScaling(X, 0, 1)
# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=2)

cls = linear_model.LinearRegression()
start = time.time()
cls.fit(X_train, y_train)
stop = time.time()
prediction = cls.predict(X_test)

print(f"Training time: {stop - start}s")


print('Intercept of linear regression model', cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
print('Accuracy ',metrics.r2_score(y_test,prediction))







