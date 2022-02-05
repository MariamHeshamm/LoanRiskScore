from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle

pd.options.mode.chained_assignment = None

def handling_null_values_with_median(dataframe):
    dataframe = dataframe.fillna(dataframe.median())
    median_value=dataframe.median()

    return median_value,dataframe


def handling_null_values_with_mean(dataframe):
    dataframe = dataframe.fillna(dataframe.mean())
    mean_value=dataframe.mean()

    return mean_value,dataframe


def handling_null_values_with_mode(dataframe):
    dataframe = dataframe.fillna(dataframe.mode().iloc[0])
    mode_value=dataframe.mode().iloc[0]

    return mode_value,dataframe


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

def tst_preprocessing(X,Y,numeric_cols_with_outliers,categorical_col,String_cols,target_col,train_median_value,train_categorical_mode_value):
    #drop cols
    X.drop(['ListingNumber'], axis=1, inplace=True)
    X.drop(['CreditGrade'], axis=1, inplace=True)  # axis = 1 column axis = 0 rows
    X.drop(['TotalProsperPaymentsBilled'], axis=1, inplace=True)

    # change value to null in train
    to_null_cols = ['IncomeRange', 'EmploymentStatus']
    to_null_values = [['Not displayed'], ['Not available']]
    j = 0
    for i in X[to_null_cols]:
        # print(X[i].value_counts())
        temp3 = to_null(X[i], to_null_values[j])
        X[i] = temp3
        # print(X[i].value_counts())
        j += 1

    # if you want to remove outliers first
    X = remove_outliers(X, numeric_cols_with_outliers)

    # apply median to test data
    X[numeric_cols_with_outliers] = X[numeric_cols_with_outliers].fillna(train_median_value)
    # X_test[numeric_cols_with_outliers]=X_test[numeric_cols_with_outliers].fillna(train_mean_value)
    X[categorical_col] = X[categorical_col].fillna(train_categorical_mode_value)

    # apply normalization
    X[numeric_cols_with_outliers] = featureScaling(X[numeric_cols_with_outliers], 0, 1)

    # apply feature encoding
    Feature_Encoder(X, String_cols)
    Feature_Encoder(Y, target_col)

    return X, Y

#with open('median_values.pkl', 'wb') as f:
    #pickle.dump(train_median_value, f)

with open('median_values.pkl', 'rb') as f:
   train_median_value = pickle.load(f)

#with open('mode_values.pkl', 'wb') as f:
    #pickle.dump(train_categorical_mode_value, f)

with open('mode_values.pkl', 'rb') as f:
    train_categorical_mode_value = pickle.load(f)



data2= pd.read_csv('LoanRiskTestForTAsClassification.csv')
test_df = pd.DataFrame(data2)
valued_rows_in_test = test_df[test_df['ProsperRating (Alpha)'].notna()]
valued_rows_in_test=valued_rows_in_test.reset_index(drop=True)

X = valued_rows_in_test.iloc[:, 0:23]
Y = valued_rows_in_test['ProsperRating (Alpha)']

Y=Y.to_frame()
numeric_cols_with_outliers = ["BorrowerAPR", "EmploymentStatusDuration", "CreditScoreRangeLower","CreditScoreRangeUpper",
                              "RevolvingCreditBalance", "BankcardUtilization", "AvailableBankcardCredit", "TotalTrades",
                              "DebtToIncomeRatio", "BorrowerRate", "StatedMonthlyIncome", "LoanOriginalAmount","LoanNumber"]
String_cols = ['LoanStatus','BorrowerState','EmploymentStatus','IsBorrowerHomeowner','IncomeRange']
target_col=["ProsperRating (Alpha)"]

categorical_col=['Term','ListingCategory (numeric)','BorrowerState','EmploymentStatus','IsBorrowerHomeowner','LoanStatus','IncomeRange']



X,Y=tst_preprocessing(X,Y,numeric_cols_with_outliers,categorical_col,String_cols,target_col,train_median_value,train_categorical_mode_value)

Pkl_Filename = "Pickle_RL_Model.pkl"
with open(Pkl_Filename, 'rb') as file:
    Pickled_LR_Model = pickle.load(file)

# Calculate the Score
score = Pickled_LR_Model.score(X, Y)
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(X)
print("Y_predict: ",Ypredict)


