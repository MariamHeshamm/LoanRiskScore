
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import time
# %%

dataframe = pd.read_csv('LoanRiskClassification.csv')
print(dataframe.shape)  # (107937, 24)

# %%

pd.options.mode.chained_assignment = None


def handling_null_values_with_median(dataframe):
    dataframe = dataframe.fillna(dataframe.median())
    median_value = dataframe.median()

    return median_value, dataframe


def handling_null_values_with_mean(dataframe):
    dataframe = dataframe.fillna(dataframe.mean())
    mean_value = dataframe.mean()

    return mean_value, dataframe


def handling_null_values_with_mode(dataframe):
    dataframe = dataframe.fillna(dataframe.mode().iloc[0])
    mode_value = dataframe.mode().iloc[0]

    return mode_value, dataframe


def to_null(dataframe, value):
    # print(len(dataframe))
    for i in range(dataframe.shape[0]):
        for j in value:
            if dataframe[i] == j:
                dataframe[i] = np.nan
    return dataframe


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def FeatureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1] - 1))
    for i in range(X.shape[1] - 1):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def one_hot_encoder_loanstatus(df):
    newdf = df
    label_encoder = LabelEncoder()
    # Encode labels in column 'Country'.
    df['LoanStatus'] = label_encoder.fit_transform(df['LoanStatus'])
    # one hot encoder

    onehotencoder = OneHotEncoder()
    x = onehotencoder.fit_transform(df.LoanStatus.values.reshape(-1, 1)).toarray()

    dfOneHot = pd.DataFrame(x, columns=["LoanStatus_" + str(int(i)) for i in range(11)])
    newdf = pd.concat([df, dfOneHot], axis=1)
    newdf = newdf.drop(['LoanStatus'], axis=1)

    newdf = newdf.drop(['LoanStatus_10'], axis=1)

    return newdf


def one_hot_encoder_employmentstatus(df):
    newdf = df
    label_encoder = LabelEncoder()
    # Encode labels in column 'Country'.
    df['EmploymentStatus'] = label_encoder.fit_transform(df['EmploymentStatus'])
    # one hot encoder

    onehotencoder = OneHotEncoder()
    x = onehotencoder.fit_transform(df.EmploymentStatus.values.reshape(-1, 1)).toarray()

    dfOneHot = pd.DataFrame(x, columns=["EmploymentStatus_" + str(int(i)) for i in range(7)])
    newdf = pd.concat([df, dfOneHot], axis=1)
    newdf = newdf.drop(['EmploymentStatus'], axis=1)

    newdf.drop(['EmploymentStatus_6'], axis=1, inplace=True)

    return newdf


def process_on_homeowner(X):
    lbl = LabelEncoder()
    lbl.fit(list(X['IsBorrowerHomeowner'].values))
    X['IsBorrowerHomeowner'] = lbl.transform(list(X['IsBorrowerHomeowner'].values))
    return X


def process_on_incomerange_withdoutdisplayed(df):
    for i in range(len(df)):

        if df.iat[i, 16] == 'Not employed':
            df.iat[i, 16] = 0

        if df.iat[i, 16] == '$0 ':
            df.iat[i, 16] = 1

        if df.iat[i, 16] == '$1-24,999':
            df.iat[i, 16] = 2

        if df.iat[i, 16] == '$25,000-49,999':
            df.iat[i, 16] = 3

        if df.iat[i, 16] == '$50,000-74,999':
            df.iat[i, 16] = 4

        if df.iat[i, 16] == '$75,000-99,999':
            df.iat[i, 16] = 5

        if df.iat[i, 16] == '$100,000+':
            df.iat[i, 16] = 6

    return df


def remove_outliers(dataframe, columns):
    # print(len(dataframe))
    for col in dataframe[columns]:
        # sns.boxplot(x=df[col])

        # plt.show()
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        # print(Q1, Q3)
        IQR = Q3 - Q1
        # print(IQR)
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        # print(lower_limit, upper_limit)

        df_no_outlier = dataframe[(dataframe[col] > lower_limit) & (dataframe[col] < upper_limit)]

        # sns.boxplot(x=df_no_outlier[col])

        # plt.show()

    return df_no_outlier


def Random_forest_classifier(no_of_trees, x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=no_of_trees)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    r_square = metrics.r2_score(y_test, y_predicted)

    return r_square


def Random_forest_regression(no_of_trees, x_train, x_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators=no_of_trees, random_state=0)
    regressor.fit(x_train, y_train)
    y_predicted = regressor.predict(x_test)  # test the output by changing values
    r_square = metrics.r2_score(y_test, y_predicted)

    return r_square


# %%


df = pd.DataFrame(dataframe)

# print(df.isna().sum())

X = df.iloc[:, 0:23]  # Features
Y = df['ProsperRating (Alpha)']  # Label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=2)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)
all_taining_data = pd.concat([X_train, Y_train], axis=1)

null_valued_rows = all_taining_data[all_taining_data['ProsperRating (Alpha)'].isnull()]

valued_rows = all_taining_data[all_taining_data['ProsperRating (Alpha)'].notna()]
valued_rows = valued_rows.reset_index(drop=True)
old_X_train = valued_rows.iloc[:, 0:23]
old_Y_train = valued_rows['ProsperRating (Alpha)']

old_Y_train = old_Y_train.to_frame()
# real_Y_test=real_Y_test.to_frame()
numeric_cols_with_outliers = ["BorrowerAPR", "EmploymentStatusDuration", "CreditScoreRangeLower",
                              "CreditScoreRangeUpper",
                              "RevolvingCreditBalance", "BankcardUtilization", "AvailableBankcardCredit", "TotalTrades",
                              "DebtToIncomeRatio", "BorrowerRate", "StatedMonthlyIncome", "LoanOriginalAmount",
                              "LoanNumber"]

null_string_cols = ["BorrowerState", "EmploymentStatus"]
String_cols = ['LoanStatus', 'BorrowerState', 'EmploymentStatus', 'IsBorrowerHomeowner', 'IncomeRange']
target_col = ["ProsperRating (Alpha)"]

categorical_col = ['Term', 'ListingCategory (numeric)', 'BorrowerState', 'EmploymentStatus', 'IsBorrowerHomeowner',
                   'LoanStatus', 'IncomeRange']
null_numeric_values = ["EmploymentStatusDuration", "DebtToIncomeRatio"]
# Preprocessing phase

# drop cols
old_X_train.drop(['ListingNumber'], axis=1, inplace=True)
old_X_train.drop(['CreditGrade'], axis=1, inplace=True)  # axis = 1 column axis = 0 rows
old_X_train.drop(['TotalProsperPaymentsBilled'], axis=1, inplace=True)

# change value to null in train
temp = to_null(old_X_train['IncomeRange'], ['Not displayed'])
old_X_train['IncomeRange'] = temp

# if you want to remove outliers first
old_X_train = remove_outliers(old_X_train, numeric_cols_with_outliers)

# get median of each feature
train_median_value, old_X_train[numeric_cols_with_outliers] = handling_null_values_with_median(
    old_X_train[numeric_cols_with_outliers])

# get mean of each feature
# train_mean_value,old_X_train[numeric_cols_with_outliers]=handling_null_values_with_mean(old_X_train[numeric_cols_with_outliers])

# get mode of each feature
train_categorical_mode_value, old_X_train[categorical_col] = handling_null_values_with_mode(
    old_X_train[categorical_col])

# apply normalization
old_X_train[numeric_cols_with_outliers] = featureScaling(old_X_train[numeric_cols_with_outliers], 0, 1)

# apply feature encoding
Feature_Encoder(old_X_train, String_cols)
Feature_Encoder(old_Y_train, target_col)

# %%

all_testing_data = pd.concat([X_test, Y_test], axis=1)

null_valued_rows_in_test = all_testing_data[all_testing_data['ProsperRating (Alpha)'].isnull()]

valued_rows_in_test = all_testing_data[all_testing_data['ProsperRating (Alpha)'].notna()]

valued_rows_in_test = valued_rows_in_test.reset_index(drop=True)
X_test = valued_rows_in_test.iloc[:, 0:23]
Y_test = valued_rows_in_test['ProsperRating (Alpha)']

Y_test = Y_test.to_frame()

X_test.drop(['ListingNumber'], axis=1, inplace=True)
X_test.drop(['CreditGrade'], axis=1, inplace=True)  # axis = 1 column axis = 0 rows
X_test.drop(['TotalProsperPaymentsBilled'], axis=1, inplace=True)

# change value to null in train
temp2 = to_null(X_test['IncomeRange'], ['Not displayed'])
X_test['IncomeRange'] = temp2

# apply median to test data
X_test[numeric_cols_with_outliers] = X_test[numeric_cols_with_outliers].fillna(train_median_value)
# X_test[numeric_cols_with_outliers]=X_test[numeric_cols_with_outliers].fillna(train_mean_value)
X_test[categorical_col] = X_test[categorical_col].fillna(train_categorical_mode_value)

# apply normalization
X_test[numeric_cols_with_outliers] = featureScaling(X_test[numeric_cols_with_outliers], 0, 1)

# apply feature encoding
Feature_Encoder(X_test, String_cols)
Feature_Encoder(Y_test, target_col)

# %%

C = 1
start = time.time()
svc = svm.SVC(kernel='linear', C=C).fit(old_X_train, old_Y_train)
stop = time.time()

predictions = np.array(svc.predict(X_test))
y_test = np.array(Y_test)
count = 0
for i in range(len(Y_test)):
    if (y_test[i] == predictions[i]):
        count += 1
accuracy = np.mean(count / len(Y_test))

print('svc', "  :  ", accuracy, " time: ", stop - start)

start = time.time()
model = BaggingClassifier(svc, n_estimators=10, random_state=0).fit(old_X_train, old_Y_train)
stop = time.time()

# %%

print(stop - start)

# %%

predictions = np.array(model.predict(X_test))
y_test = np.array(Y_test)
count = 0
for i in range(len(Y_test)):
    if (y_test[i] == predictions[i]):
        count += 1
accuracy = np.mean(count / len(Y_test))

print('baggingClassifier', "  :  ", accuracy, " time: ", stop - start)

