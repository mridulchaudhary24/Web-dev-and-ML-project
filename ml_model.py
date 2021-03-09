import pandas as pd
data = pd.read_csv("heart_3.csv")
corr_matrix = data.corr()
corr_matrix['target'].sort_values(ascending = False)

data.drop(["restecg","fbs","chol","trestbps","sex","thal"], axis = 1,inplace = True) 



#Train-test-splitting
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state= 42)
for train_index, test_index in split.split(data, data['exang']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state= 42)
for train_index, test_index in split.split(data, data['slope']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
for train_index, test_index in split.split(data, data['ca']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
for train_index, test_index in split.split(data, data['target']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

train_set = strat_train_set.copy()
test_set  = strat_test_set.copy()

#Creating pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import  numpy as np
def pass_through_pipe(dataframe):
    pipe_mean= Pipeline([('imputer' , SimpleImputer(strategy='mean')) , ('std_scaler' , StandardScaler())])
    pipe_mode= Pipeline([('imputer' , SimpleImputer(strategy='most_frequent')) , ('std_scaler' , StandardScaler())])
#     now storing the pipline raw data into numpy array
    numpy_value_mean = pipe_mean.fit_transform(dataframe[['age' , 'thalach' , 'oldpeak']])
    numpy_value_mode = pipe_mode.fit_transform(dataframe[['cp' , 'exang' , 'slope' , 'ca']])
    result = np.concatenate((numpy_value_mean , numpy_value_mode) , axis=1)
    return result

target= train_set['target']

train_set = train_set.drop(['target'] , axis=1)

features = pass_through_pipe(train_set)

from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(features , target)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(features , target)

from sklearn.ensemble import RandomForestClassifier
random_forest_model= RandomForestClassifier(n_estimators=100)
random_forest_model.fit(features,target)

from sklearn import svm
vector_model = svm.SVC(kernel='linear') 
vector_model.fit(features, target)

def calcutale_result(age,cp,thalach,exang,oldpeak,slope,ca):
    temp= train_set.copy()
    dict_input= dict()
    dict_input['age']= age
    dict_input['cp']= cp
    dict_input['thalach']= thalach
    dict_input['exang']= exang
    dict_input['oldpeak']= oldpeak
    dict_input['slope']= slope
    dict_input['ca']= ca
    temp= temp.append(dict_input, ignore_index=True)
    result_logistic= logistic_regression_model.predict(pass_through_pipe(temp))[-1]
    probablity_logistic= logistic_regression_model.predict_proba(pass_through_pipe(temp))[-1][-1]*100
    
    # result_tree= tree_model.predict(pass_through_pipe(temp))[-1]
    # probablity_tree= tree_model.predict_proba(pass_through_pipe(temp))[-1]
    
    # result_forest= random_forest_model.predict(pass_through_pipe(temp))[-1]
    # probablity_forest= random_forest_model.predict_proba(pass_through_pipe(temp))[-1]
    
    # result_svm= vector_model.predict(pass_through_pipe(temp))[-1]

    #giving advice
    if int(probablity_logistic) <=30:
        advice= "It looks like you are fine. Keep yourself healthy" 
    elif int(probablity_logistic) >30 and int(probablity_logistic)<=50:
        advice= "It looks like you are fine. However we advice you to start exercise and prevent early symptoms"   
    elif int(probablity_logistic) >50 and int(probablity_logistic)<=70:
        advice= "It looks like you should take concern now. Don't panic, you still have plenty of time. Make your lifestyle better."   
    elif int(probablity_logistic) >70 and int(probablity_logistic)<=90:
        advice= "It is time to take some serious steps for your health. You can still make it. You have high chances of heart disease however you can still change it"   
    else:
        advice= "Don't panic. We concern you to reach out your doctor and go through appropriate examinations"   
    return [ int(probablity_logistic), advice ]
    # return result_logistic
#     probablity_svm= vector_model.predict_proba(pass_through_pipe(temp))[-1]
    # print("=======logistic regression=========")
    # print("Probability= ",result_logistic)
    # print("Probability= ",probablity_logistic)
    # print("=======decision tree=========")
    # print("Probability= ",result_tree)
    # print("Probability= ",probablity_tree)
    # print("=======random forest=========")
    # print("Probability= ",result_forest)
    # print("Probability= ",probablity_forest)
    # print("=======SVM=========")
    # print("Probability= ",result_svm)
#     print("Probability= ",probablity_svm)
#     if result==1:
#         print("You might suffer from heart disease")
#     else:
#         print("Take a chill pill. You are good")