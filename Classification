import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
import statsmodels.api as sm
from scipy.stats import boxcox
from scipy import stats
import pylab
from sklearn.model_selection import cross_val_score, StratifiedKFold

test = pd.read_csv('HE/Pet_adoption/test.csv')
train = pd.read_csv('HE/Pet_adoption/train.csv')
desc = train.describe()
print(train.info())
print(train.columns)


# print(train.isnull().sum())
# plt.figure(figsize = (12,10))
# sns.heatmap(train.isnull(),cmap='gray')


# print(test.isnull().sum())
# plt.figure(figsize = (12,10))
# sns.heatmap(test.isnull(),cmap='gray')


sasa = train.groupby('condition').mean()
# Creating a new condition
train['condition'] = train['condition'].fillna(int(5))
test['condition'] = test['condition'].fillna(int(5))


# print(train.isnull().sum())
# plt.figure(figsize = (12,10))
# sns.heatmap(train.isnull(),cmap='gray')


# print(test.isnull().sum())
# plt.figure(figsize = (12,10))
# sns.heatmap(test.isnull(),cmap='gray')


# plt.figure(figsize = (15,10))
# sns.pairplot(train)


# plt.figure(figsize = (15,10))
# sns.heatmap(train.corr(), annot = True)



train.loc[train['listing_date'] < train['issue_date']] #2 Anamolies
test.loc[test['listing_date'] < test['issue_date']] #0 Anamolies
train = train.loc[train['listing_date'] >= train['issue_date']]
# test = test.loc[test['listing_date'] >= test['issue_date']]


train['issue_date_Year'] = train['issue_date'].apply(lambda x: int(x[:4]))
train['issue_date_Month'] = pd.to_datetime(train['issue_date']).dt.month
train['issue_date_Day'] = pd.to_datetime(train['issue_date']).dt.day
train['issue_date_Dayofweek'] = pd.to_datetime(train['issue_date']).dt.dayofweek
train['issue_date_DayOfyear'] = pd.to_datetime(train['issue_date']).dt.dayofyear
train['issue_date_Week_No'] = pd.to_datetime(train['issue_date']).dt.week
train['issue_date_Quarter'] = pd.to_datetime(train['issue_date']).dt.quarter 
train['issue_date_Is_month_start'] = pd.to_datetime(train['issue_date']).dt.is_month_start
train['issue_date_Is_month_end'] = pd.to_datetime(train['issue_date']).dt.is_month_end
train['issue_date_Is_quarter_start'] = pd.to_datetime(train['issue_date']).dt.is_quarter_start
train['issue_date_Is_quarter_end'] = pd.to_datetime(train['issue_date']).dt.is_quarter_end
train['issue_date_Is_year_start'] = pd.to_datetime(train['issue_date']).dt.is_year_start
train['issue_date_Is_year_end'] = pd.to_datetime(train['issue_date']).dt.is_year_end
train['issue_date_Is_weekend'] = np.where(train['issue_date_Dayofweek'].isin([5,6]),1,0)
train['issue_date_Is_weekday'] = np.where(train['issue_date_Dayofweek'].isin([0,1,2,3,4]),1,0)

test['issue_date_Year'] = test['issue_date'].apply(lambda x: int(x[:4]))
test['issue_date_Month'] = pd.to_datetime(test['issue_date']).dt.month
test['issue_date_Day'] = pd.to_datetime(test['issue_date']).dt.day
test['issue_date_Dayofweek'] = pd.to_datetime(test['issue_date']).dt.dayofweek
test['issue_date_DayOfyear'] = pd.to_datetime(test['issue_date']).dt.dayofyear
test['issue_date_Week_No'] = pd.to_datetime(test['issue_date']).dt.week
test['issue_date_Quarter'] = pd.to_datetime(test['issue_date']).dt.quarter 
test['issue_date_Is_month_start'] = pd.to_datetime(test['issue_date']).dt.is_month_start
test['issue_date_Is_month_end'] = pd.to_datetime(test['issue_date']).dt.is_month_end
test['issue_date_Is_quarter_start'] = pd.to_datetime(test['issue_date']).dt.is_quarter_start
test['issue_date_Is_quarter_end'] = pd.to_datetime(test['issue_date']).dt.is_quarter_end
test['issue_date_Is_year_start'] = pd.to_datetime(test['issue_date']).dt.is_year_start
test['issue_date_Is_year_end'] = pd.to_datetime(test['issue_date']).dt.is_year_end
test['issue_date_Is_weekend'] = np.where(test['issue_date_Dayofweek'].isin([5,6]),1,0)
test['issue_date_Is_weekday'] = np.where(test['issue_date_Dayofweek'].isin([0,1,2,3,4]),1,0)
    


train['listing_date_Year'] = train['listing_date'].apply(lambda x: int(x[:4]))
train['listing_date_Month'] = pd.to_datetime(train['listing_date']).dt.month
train['listing_date_Day'] = pd.to_datetime(train['listing_date']).dt.day
train['listing_date_Dayofweek'] = pd.to_datetime(train['listing_date']).dt.dayofweek
train['listing_date_DayOfyear'] = pd.to_datetime(train['listing_date']).dt.dayofyear
train['listing_date_Week_No'] = pd.to_datetime(train['listing_date']).dt.week
train['listing_date_Quarter'] = pd.to_datetime(train['listing_date']).dt.quarter 
train['listing_date_Is_month_start'] = pd.to_datetime(train['listing_date']).dt.is_month_start
train['listing_date_Is_month_end'] = pd.to_datetime(train['listing_date']).dt.is_month_end
train['listing_date_Is_quarter_start'] = pd.to_datetime(train['listing_date']).dt.is_quarter_start
train['listing_date_Is_quarter_end'] = pd.to_datetime(train['listing_date']).dt.is_quarter_end
train['listing_date_Is_year_start'] = pd.to_datetime(train['listing_date']).dt.is_year_start
train['listing_date_Is_year_end'] = pd.to_datetime(train['listing_date']).dt.is_year_end
train['listing_date_Is_weekend'] = np.where(train['issue_date_Dayofweek'].isin([5,6]),1,0)
train['listing_date_Is_weekday'] = np.where(train['issue_date_Dayofweek'].isin([0,1,2,3,4]),1,0)

test['listing_date_Year'] = test['listing_date'].apply(lambda x: int(x[:4]))
test['listing_date_Month'] = pd.to_datetime(test['listing_date']).dt.month
test['listing_date_Day'] = pd.to_datetime(test['listing_date']).dt.day
test['listing_date_Dayofweek'] = pd.to_datetime(test['listing_date']).dt.dayofweek
test['listing_date_DayOfyear'] = pd.to_datetime(test['listing_date']).dt.dayofyear
test['listing_date_Week_No'] = pd.to_datetime(test['listing_date']).dt.week
test['listing_date_Quarter'] = pd.to_datetime(test['listing_date']).dt.quarter 
test['listing_date_Is_month_start'] = pd.to_datetime(test['listing_date']).dt.is_month_start
test['listing_date_Is_month_end'] = pd.to_datetime(test['listing_date']).dt.is_month_end
test['listing_date_Is_quarter_start'] = pd.to_datetime(test['listing_date']).dt.is_quarter_start
test['listing_date_Is_quarter_end'] = pd.to_datetime(test['listing_date']).dt.is_quarter_end
test['listing_date_Is_year_start'] = pd.to_datetime(test['listing_date']).dt.is_year_start
test['listing_date_Is_year_end'] = pd.to_datetime(test['listing_date']).dt.is_year_end
test['listing_date_Is_weekend'] = np.where(test['issue_date_Dayofweek'].isin([5,6]),1,0)
test['listing_date_Is_weekday'] = np.where(test['issue_date_Dayofweek'].isin([0,1,2,3,4]),1,0)


train['diff_date']= train['listing_date_Day'] - train['issue_date_Day']
test['diff_date']= test['listing_date_Day'] - test['issue_date_Day']

train['diff_month']= train['listing_date_Month'] - train['issue_date_Month']
test['diff_month']= test['listing_date_Month'] - test['issue_date_Month']

train['diff_year'] = train['listing_date_Year'] - train['issue_date_Year']
test['diff_year'] = test['listing_date_Year'] - test['issue_date_Year']


train.drop(['issue_date','listing_date'], axis = 1, inplace=True)
test.drop(['issue_date','listing_date'], axis = 1, inplace=True)





train['length(cm)'] = train['length(m)']*100
test['length(cm)'] = test['length(m)']*100
train.drop(['length(m)'], axis = 1, inplace=True)
test.drop(['length(m)'], axis = 1, inplace=True)


print(train.describe())
print(test.describe())

train[['length(cm)','height(cm)']].boxplot()
test[['length(cm)','height(cm)']].boxplot()

print(len(train[train['length(cm)'] == 0]))
print(len(test[test['length(cm)'] == 0]))


val1 = train['length(cm)'].mean()
val2 = test['length(cm)'].mean()
train['length(cm)'] = train['length(cm)'].replace(to_replace=0, value=val1)
test['length(cm)'] = test['length(cm)'].replace(to_replace=0, value=val2)




def calc_len(x):
    if x <= 20:
        return 201
    elif x <= 40:
        return 301
    elif x <= 60:
        return 401
    elif x <= 80:
        return 501
    elif x <= 100:
        return 601
        
train['length(cm)_group'] = pd.Series()
test['length(cm)_group'] = pd.Series()
train['length(cm)_group'] = train['length(cm)'].apply(lambda x: calc_len(x))
test['length(cm)_group'] = test['length(cm)'].apply(lambda x: calc_len(x))

def calc_ht(x):
    if x <= 15:
        return 61
    elif x <= 25:
        return 62
    elif x <= 35:
        return 63
    elif x <= 45:
        return 64
    elif x <= 55:
        return 65

train['height(cm)_group'] = pd.Series()
test['height(cm)_group'] = pd.Series()
train['height(cm)_group'] = train['height(cm)'].apply(lambda x: calc_ht(x))
test['height(cm)_group'] = test['height(cm)'].apply(lambda x: calc_ht(x))


train['ratio_len_height'] = train['length(cm)']/train['height(cm)']
test['ratio_len_height'] = test['length(cm)']/test['height(cm)']

train['ratio_x_height'] = train['length(cm)']*train['height(cm)']
test['ratio_x_height'] = test['length(cm)']*test['height(cm)']

# train.drop(['length(cm)'], axis = 1, inplace=True)
# test.drop(['length(cm)'], axis = 1, inplace=True)

# train.drop(['height(cm)'], axis = 1, inplace=True)
# test.drop(['height(cm)'], axis = 1, inplace=True)

# This is an example of Data Leakage
# a = dict(train.groupby('height(cm)_group').mean()['pet_category'])
# train['height(cm)_group_pet'] = train['height(cm)_group'].map(a)
# b = dict(train.groupby('height(cm)_group').mean()['pet_category'])
# test['height(cm)_group_pet'] = test['height(cm)_group'].map(b)
# train.fillna(1.7084990991431717,inplace = True)
# test.fillna(1.7084990991431717,inplace = True)

# a = dict(train.groupby('height(cm)').mean()['breed_category'])
# train['height(cm)_breed'] = train['height(cm)'].map(a)
# b = dict(train.groupby('height(cm)').mean()['breed_category'])
# test['height(cm)_breed'] = test['height(cm)'].map(b)
# train.fillna(0.5992458727712973,inplace = True)
# test.fillna(0.5992458727712973,inplace = True)

# a = dict(train.groupby('length(m)').mean()['pet_category'])
# train['length(m)_pet'] = train['length(m)'].map(a)
# b = dict(train.groupby('length(m)').mean()['pet_category'])
# test['length(m)_pet'] = test['length(m)'].map(b)
# train.fillna(1.7092466941452507,inplace = True)
# test.fillna(1.7092466941452507,inplace = True)

# a = dict(train.groupby('length(m)').mean()['breed_category'])
# train['length(m)_breed'] = train['length(m)'].map(a)
# b = dict(train.groupby('length(m)').mean()['breed_category'])
# test['length(m)_breed'] = test['length(m)'].map(b)
# train.fillna(0.6006510071357669,inplace = True)
# test.fillna(0.6006510071357669,inplace = True)


zz = ['X1','X2','issue_date_Year','issue_date_Month','issue_date_Day',
      'issue_date_DayOfyear','issue_date_Week_No',
      'listing_date_Year','listing_date_Month','listing_date_Day',
      'listing_date_DayOfyear',
      'listing_date_Week_No']

# for i in zz:
#     plt.figure(figsize =(6,4)) 
#     ax = sns.boxplot(train[i])

# for i in zz:
#     plt.figure(figsize =(6,4)) 
#     ax = sns.distplot(train[i], hist=True, color="red", kde = True)

# for i in zz:
#     plt.figure(figsize=(8,6))
#     stats.probplot(train[i], plot=pylab, dist="norm")
#     pylab.show()
#     print(i,train[i].skew(),train[i].kurt())

# for i in zz:
#     train[i] = np.log1p(train[i])
#     test[i] = np.log1p(test[i])    

# # train['length(cm)'] = np.sqrt(train['length(cm)'])
# # test['length(cm)'] = np.sqrt(test['length(cm)'])

# # train['height(cm)'] = np.sqrt(train['height(cm)'])
# # test['height(cm)'] = np.sqrt(test['height(cm)'])

# train['diff_year'] = np.sqrt(train['diff_year'])
# test['diff_year'] = np.sqrt(test['diff_year'])

# train['ratio_len_height'] = np.sqrt(train['ratio_len_height'])
# test['ratio_len_height'] = np.sqrt(test['ratio_len_height'])


# train['ratio_x_height'] = np.sqrt(train['ratio_x_height'])
# test['ratio_x_height'] = np.sqrt(test['ratio_x_height'])

zzz = []
for i in train.columns:
    if i in zz:
        continue
    elif i not in zz:
        zzz.append(i)
zzz.pop(3) 
zzz.pop(3) 
zzz.pop(3) 
zzz.pop(23) 
zzz.pop(23) 
zzz.pop(23)
zzz.pop(23)
zzz.pop(25)
zzz.pop(25)


pet_Cat = dict(train.groupby('color_type').mean()['pet_category'])
train['color_pet_cat'] = train['color_type'].map(pet_Cat)
pet_Cat2 = dict(train.groupby('color_type').mean()['pet_category'])
test['color_pet_cat'] = test['color_type'].map(pet_Cat2)

breed_Cat = dict(train.groupby('color_type').mean()['breed_category'])
train['color_breed_Cat'] = train['color_type'].map(breed_Cat)
breed_Cat2 = dict(train.groupby('color_type').mean()['breed_category'])
test['color_breed_Cat'] = test['color_type'].map(breed_Cat2)

print(train['color_type'].nunique())
print(test['color_type'].nunique())

set(train.color_type) - set(test.color_type)

# from sklearn.preprocessing import LabelEncoder

# train['color_type_Enc'] = train['color_type'].apply(LabelEncoder().fit_transform)
# test['color_type_Enc'] = test['color_type'].apply(LabelEncoder().fit_transform)

train.drop(['color_type'], axis = 1, inplace=True)
test.drop(['color_type'], axis = 1, inplace=True)

zzz.pop(2) #color_type



# #Oversampling
# asas = train[train['breed_category'] == 2]
# train = pd.concat([train,asas,asas,asas,asas,asas], axis =0)
# asasas = train[train['pet_category'] == 0]
# for i in range(1,28):
#     train = pd.concat([train,asasas], axis =0)
# asasasas = train[train['pet_category'] == 4]
# train = pd.concat([train,asasasas], axis =0)
# #Its not giving good accuracy so commented 





train['pet_id_1'] = train['pet_id'].apply(lambda x: int(x[5:6]))
train['pet_id_2'] = train['pet_id'].apply(lambda x: int(x[5:7]))

# train['pet_id_3'] = train['pet_id'].apply(lambda x: int(x[5:])) This is and more id digits are very less imp   

test['pet_id_1'] = test['pet_id'].apply(lambda x: int(x[5:6]))
test['pet_id_2'] = test['pet_id'].apply(lambda x: int(x[5:7]))


pet_Cat = dict(train.groupby('pet_id_1').mean()['pet_category'])
train['color_pet_id1'] = train['pet_id_1'].map(pet_Cat)
pet_Cat2 = dict(train.groupby('pet_id_1').mean()['pet_category'])
test['color_pet_id1'] = test['pet_id_1'].map(pet_Cat2)

breed_Cat = dict(train.groupby('pet_id_1').mean()['breed_category'])
train['color_breed_id1'] = train['pet_id_1'].map(breed_Cat)
breed_Cat2 = dict(train.groupby('pet_id_1').mean()['breed_category'])
test['color_breed_id1'] = test['pet_id_1'].map(breed_Cat2)


pet_Cat = dict(train.groupby('pet_id_2').mean()['pet_category'])
train['color_pet_id2'] = train['pet_id_2'].map(pet_Cat)
pet_Cat2 = dict(train.groupby('pet_id_2').mean()['pet_category'])
test['color_pet_id2'] = test['pet_id_2'].map(pet_Cat2)

breed_Cat = dict(train.groupby('pet_id_2').mean()['breed_category'])
train['color_breed_id2'] = train['pet_id_2'].map(breed_Cat)
breed_Cat2 = dict(train.groupby('pet_id_2').mean()['breed_category'])
test['color_breed_id2'] = test['pet_id_2'].map(breed_Cat2)



train.drop(['pet_id', 'pet_id_2'], axis = 1, inplace=True)
test.drop(['pet_id', 'pet_id_2'], axis = 1, inplace=True)


zzz.append('pet_id_1')
zzz.pop(0)
# zzz.append( 'height(cm)_group')



np.random.seed(42)
y = train['pet_category']
X = train.drop(['pet_category','breed_category'], axis=1)

from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(f_classif, k=20).fit_transform(X, y)

# feats = ['condition','X1','issue_date_Year','issue_date_Month','issue_date_DayOfyear',
#         'issue_date_Week_No','issue_date_Quarter','listing_date_Month',
#         'listing_date_DayOfyear','listing_date_Week_No','listing_date_Quarter',
#         'diff_month', 'diff_year','color_pet_cat', 'color_breed_Cat']

feats = ['condition','X1','X2','issue_date_Year', 'issue_date_Month', 'issue_date_Day',
'issue_date_DayOfyear', 'issue_date_Week_No', 'issue_date_Quarter',
'listing_date_Year', 'listing_date_Month','listing_date_Week_No', 'listing_date_Quarter',
'diff_date', 'diff_month', 'diff_year','color_pet_cat', 'color_breed_Cat',
'color_pet_id2', 'color_breed_id2']



trains = pd.concat([train[feats] ,y], axis= 1)
tests = test[feats]

# trains = trains[trains['issue_date_Year']>=2000]
# trains = trains[trains['listing_date_Year']>2015]

# for i in trains.columns:
#     plt.figure(figsize=(10,8))
#     sns.boxplot(trains[i])

# for i in r:
#     plt.figure(figsize=(10,8))
#     sns.distplot(trains[i])

s = ['condition','issue_date_Month','listing_date_Quarter',
      'issue_date_Quarter','listing_date_Month']

r = ['X1', 'X2','issue_date_Year', 'issue_date_Day', 'issue_date_DayOfyear',
      'issue_date_Week_No', 'listing_date_Year', 'listing_date_Week_No']
     
for i in r:
    trains[i] = np.sqrt(trains[i])
    tests[i] = np.sqrt(tests[i]) 

# for i in ['diff_date','diff_month','diff_year']:
#     trains[i] = np.sqrt(trains[i])
#     tests[i] = np.sqrt(tests[i]) 

ww = []
for i in s:
    ww.append(trains[i].value_counts().count())

df = pd.concat([trains,tests], axis= 0)
for i in s:
    dummy = pd.get_dummies(df[i], drop_first=True, prefix = i)
    df = pd.concat([df,dummy], axis= 1)
    df.drop([i], inplace=True, axis= 1)

trains = df[:18832]
tests = df[18832:]
tests.drop(['pet_category'], axis=1, inplace=True)







skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
def cv_f1_w(model):
    f1_w = cross_val_score(model, X2_train, y_train.values.ravel(), scoring="f1_sweighted", 
                           cv=skf, n_jobs=-1, verbose = True)
    return (f1_w)





np.random.seed(42)
y = train['pet_category']
X = train.drop(['pet_category'], axis=1)

# # Oversampling with SMOTETomek
# from imblearn.combine import SMOTETomek
# smt = SMOTETomek(sampling_strategy = 'minority', random_state=1996)
# X_res, y_res = smt.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25, random_state=42)


sc_X = RobustScaler()
X2_train = sc_X.fit_transform(X_train)
X2_test = sc_X.fit_transform(X_test)
tests = sc_X.fit_transform(test)

# X_sm = sm.add_constant(X)
# model = sm.OLS(y,X_sm)
# print(model.fit().summary())




from sklearn.ensemble import ExtraTreesClassifier

ss = ExtraTreesClassifier()
print(cv_f1_w(ss).mean())
ss.fit(X2_train, y_train)
# feats = pd.Series(ss.feature_importances_ * 100)
ss_pred = ss.predict(X2_test)
print('ExtraTreesClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, ss_pred, average='weighted'))
print(classification_report(y_test,ss_pred))
print(confusion_matrix(y_test, ss_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, ss_pred),annot = True,cmap='BrBG')
plt.show()





from sklearn.naive_bayes import GaussianNB
print('with non-scaled features:')
nb = GaussianNB()
print(cv_f1_w(nb).mean())
nb.fit(X2_train,y_train)
nb_pred = nb.predict(X2_test)
print('GaussianNB Performance:')
print('f1_score:', metrics.f1_score(y_test, nb_pred, average='weighted'))
print(classification_report(y_test,nb_pred))
print(confusion_matrix(y_test, nb_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, nb_pred),annot = True,cmap='BrBG')
plt.show()



from sklearn.neural_network import MLPClassifier
print('with non-scaled features:')
mlp = MLPClassifier()
print(cv_f1_w(mlp).mean())
mlp.fit(X2_train,y_train)
mlp_pred = mlp.predict(X2_test)
print('MLPClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, mlp_pred, average='weighted'))
print(classification_report(y_test,mlp_pred))
print(confusion_matrix(y_test, mlp_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, mlp_pred),annot = True,cmap='BrBG')
plt.show()




print('with non-scaled features:')
lm1 = LogisticRegression(C=10000000.0, penalty='none', solver='saga')
print(cv_f1_w(lm1).mean())
#LogisticRegression(C=10.0, solver='liblinear', penalty = 'l2')
lm1.fit(X2_train,y_train)
lm1_pred = lm1.predict(X2_test)
print('Logistic Regression Performance:')
print('f1_score:', metrics.f1_score(y_test, lm1_pred, average='weighted'))
print(classification_report(y_test,lm1_pred))
print(confusion_matrix(y_test, lm1_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, lm1_pred),annot = True,cmap='BrBG')
plt.show()


# param_grid = {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
#     'C' : 10.0**np.arange(-6,8), 
#     'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
# clf = GridSearchCV(estimator = LogisticRegression(), param_grid = param_grid, refit = True,
#                     cv = 3, verbose=True, n_jobs=-1, scoring = 'f1_weighted')
# clf.fit(X2_train,y_train)
# print(clf.best_params_)
# print(clf.best_estimator_)




print('\nwith non-scaled features:')
DT = DecisionTreeClassifier()
print(cv_f1_w(DT).mean())
DT.fit(X2_train, y_train)
DT_pred = DT.predict(X2_test)
print('DecisionTreeClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, DT_pred, average='weighted'))
print(classification_report(y_test,DT_pred))
print(confusion_matrix(y_test, DT_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, DT_pred),annot = True,cmap='BrBG')
plt.show()




print('\nwith non-scaled features:')
ada = AdaBoostClassifier()
print(cv_f1_w(ada).mean())
ada.fit(X2_train, y_train)
ada_pred = ada.predict(X2_test)
print('AdaBoostClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, ada_pred, average='weighted'))
print(classification_report(y_test,ada_pred))
print(confusion_matrix(y_test, ada_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, ada_pred),annot = True,cmap='BrBG')
plt.show()




print('with non-scaled features:')
rf = RandomForestClassifier()#max_features='sqrt', n_estimators=300,
                            # class_weight = dict([{0: 1, 1: 2000000}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 5000}]))
print(cv_f1_w(rf).mean())
rf.fit(X2_train,y_train)
rf_pred = rf.predict(X2_test)
print('RandomForestClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, rf_pred, average='weighted'))
print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test, rf_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, rf_pred),annot = True,cmap='BrBG')
plt.show()

# rf_param_grid = {'max_features': ['sqrt', 'auto','log2'],
#               'min_samples_leaf': [1, 3, 5, 8],
#               'n_estimators': [100, 300, 500, 800, 1500]}
# rf_grid = GridSearchCV(estimator= RandomForestClassifier(), param_grid = rf_param_grid, refit = True
#                         ,scoring='f1_weighted' , n_jobs=-1, verbose=True, cv = 3)
# rf_grid.fit(X2_train,y_train)
# print(rf_grid.best_params_)
# print(rf_grid.best_estimator_)




print('with non-scaled features:')
svr = SVC(probability = False)#(C=10, gamma=0.01, probability = False)
print(cv_f1_w(svr).mean())
svr.fit(X2_train,y_train)
svr_pred = svr.predict(X2_test)
print('SVC Performance:')
print('f1_score:', metrics.f1_score(y_test, svr_pred, average='weighted'))
print(classification_report(y_test,svr_pred))
print(confusion_matrix(y_test, svr_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, svr_pred),annot = True,cmap='BrBG')
plt.show()

# param_grid = {'C': [1, 10, 25, 80, 100], 'gamma': [0.1, 0.01,0.001,0.0001]}#, 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']} 
# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=True,cv=3,n_jobs=-1,scoring='f1_weighted')
# grid.fit(X2_train,y_train)
# print(grid.best_params_)
# print(grid.best_estimator_)




print('with non-scaled features:')
xbgc = xgb.XGBClassifier()
print(cv_f1_w(xbgc).mean())
xbgc.fit(X2_train,y_train)
xbgc_pred = xbgc.predict(X2_test)
print('XGB Performance:')
print('f1_score:', metrics.f1_score(y_test, xbgc_pred, average='weighted'))
print(classification_report(y_test,xbgc_pred))
print(confusion_matrix(y_test, xbgc_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, xbgc_pred),annot = True,cmap='BrBG')
plt.show()

# params = {"learning_rate"    : [  0.1, 0.01,  0.001],
#   "max_depth"        : [  3, 5, 8],
#   "min_child_weight" : [  3, 5, 8 ],
#   "colsample_bytree" : [ 0.1, 0.3, 0.5],
#   'n_estimators': [ 100, 300, 500, 800, 1500]}
# xgb_tuning = GridSearchCV(estimator = xgb.XGBClassifier(random_state=1996),
#                           param_grid = params, refit = True
#                           n_jobs=-1, scoring = 'f1_weighted',
#                           cv=3, verbose=True)
# xgb_tuning.fit(X2_train,y_train)
# print(xgb_tuning.best_params_)
# print(xgb_tuning.best_estimator_)




print('\nwith non-scaled features:')
gbm1 = GradientBoostingClassifier(random_state=42)
print(cv_f1_w(gbm1).mean())
gbm1.fit(X2_train, y_train)
gbm1_pred = gbm1.predict(X2_test)
print('Gradiant Boosting Performance:')
print('f1_score:', metrics.f1_score(y_test, gbm1_pred, average='weighted'))
print(classification_report(y_test,gbm1_pred))
print(confusion_matrix(y_test, gbm1_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, gbm1_pred),annot = True,cmap='BrBG')
plt.show()

# gbm_param_grid = {'learning_rate':[0.1, 0.01, 0.001], 
#             'n_estimators':[100, 300, 500, 1000],
#           'max_depth':[3, 5, 8],
#           'min_samples_split':[10, 20],
#           'max_features':[4, 7]}
# gbm_tuning = GridSearchCV(estimator =GradientBoostingClassifier(random_state=1996),
#                           param_grid = gbm_param_grid, verbose = True,
#                           n_jobs=-1, scoring='f1_weighted',
#                           cv=3, refit = True)
# gbm_tuning.fit(X_train,y_train)
# print(gbm_tuning.best_params_)
# print(gbm_tuning.best_estimator_)




print('\nwith non-scaled features:')
knn = KNeighborsClassifier(weights = 'distance')
print(cv_f1_w(knn).mean())
knn.fit(X2_train, y_train)
knn_pred = knn.predict(X2_test)
print('KNeighborsClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, knn_pred, average='weighted'))
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test, knn_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, knn_pred),annot = True,cmap='BrBG')
plt.show()

# E = []
# for i in range(1,51):
#     knn = KNeighborsClassifier(n_neighbors=i,weights = 'distance')
#     knn.fit(X_train, y_train)
#     knn_preds = knn.predict(X_test)
#     E.append(metrics.f1_score(y_test, knn_preds, average='weighted'))

# fig = plt.figure(figsize=(10,8))    
# plt.plot(range(1,51), E, ls = '--',lw = 2 , markersize = 4, marker = 'o', markerfacecolor = 'red',color = 'blue')
# plt.grid()


from sklearn.ensemble import BaggingClassifier
print('\nwith non-scaled features:')
bgc = BaggingClassifier()
print(cv_f1_w(bgc).mean())
bgc.fit(X2_train, y_train)
bgc_pred = bgc.predict(X2_test)
print('BaggingClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, bgc_pred, average='weighted'))
print(classification_report(y_test,bgc_pred))
print(confusion_matrix(y_test, bgc_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, bgc_pred),annot = True,cmap='BrBG')
plt.show()




print('\nwith non-scaled features:')
lgbc = lgb.LGBMClassifier()
print(cv_f1_w(lgbc).mean())
lgbc.fit(X2_train, y_train)
lgbc_pred = lgbc.predict(X2_test)
print('LGBMClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, lgbc_pred, average='weighted'))
print(classification_report(y_test,lgbc_pred))
print(confusion_matrix(y_test, lgbc_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, lgbc_pred),annot = True,cmap='BrBG')
plt.show()



# print('with non-scaled features:')
# cbr1 = cb.CatBoostClassifier()
# print(cv_f1_w(cbr1).mean())
# cbr1.fit(X2_train, y_train)
# cbr1_pred = cbr1.predict(X2_test)
# print('CatBoostClassifier Performance:')
# print('f1_score:', metrics.f1_score(y_test, cbr1_pred, average='weighted'))
# print(classification_report(y_test,cbr1_pred))
# print(confusion_matrix(y_test, cbr1_pred))
# fig = plt.figure(figsize=(10,8))
# sns.heatmap(confusion_matrix(y_test, cbr1_pred),annot = True,cmap='BrBG')
# plt.show()


from sklearn.ensemble import VotingClassifier
vclf = VotingClassifier([('xbgc',xbgc),('bgc',bgc),('ss',ss),
                         ('rf',rf),('lgbc',lgbc),('gbm1',gbm1)],voting='soft')
print(cv_f1_w(vclf).mean())
vclf.fit(X2_train, y_train)
vclf_pred = vclf.predict(X2_test)
vclf_pred_prob = vclf.predict_proba(X2_test)
print('VotingClassifier Performance:')
print('roc_auc_score:', metrics.roc_auc_score(y_test, vclf_pred_prob, multi_class = 'ovr'))
print('f1_score:', metrics.f1_score(y_test, vclf_pred, average = 'weighted'))
print('accuracy_score:', metrics.accuracy_score(y_test, vclf_pred))
print(classification_report(y_test,vclf_pred))
print(confusion_matrix(y_test, vclf_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, vclf_pred),annot = True,cmap='BrBG')
plt.show()





np.random.seed(42)
y = train['breed_category']
X = train.drop(['pet_category','breed_category'], axis=1)

from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(f_classif, k=20).fit_transform(X, y)


featt = ['condition','X1','X2','issue_date_Year', 'issue_date_Dayofweek',
          'issue_date_Is_weekend', 'issue_date_Is_weekday',
          'listing_date_Year', 'listing_date_Month',
          'listing_date_DayOfyear', 'listing_date_Week_No',
          'listing_date_Quarter', 'listing_date_Is_weekend',
          'listing_date_Is_weekday', 'diff_month', 'diff_year',
          'color_pet_cat', 'color_breed_Cat', 'color_pet_id1', 'color_breed_id2']
traint = pd.concat([train[featt] ,y], axis= 1)
testt = test[featt]

# # for i in traint.columns:
# #     plt.figure(figsize=(10,8))
# #     sns.boxplot(traint[i])

# # for i in trains.columns:
# #     plt.figure(figsize=(10,8))
# #     sns.distplot(trains['diff_date'])

s = ['condition','issue_date_Dayofweek','issue_date_Is_weekend', 'issue_date_Is_weekday',
      'listing_date_Month','listing_date_Quarter', 'listing_date_Is_weekend',
      'listing_date_Is_weekday',]

r = ['X1', 'X2','issue_date_Year', 'listing_date_Year',
      'listing_date_DayOfyear', 'listing_date_Week_No']

     
for i in r:
    traint[i] = np.sqrt(traint[i])
    testt[i] = np.sqrt(testt[i]) 

for i in ['diff_year']:
    traint[i] = np.log1p(traint[i])
    testt[i] = np.log1p(testt[i]) 

ww = []
for i in s:
    ww.append(traint[i].value_counts().count())

df = pd.concat([traint,testt], axis= 0)
for i in s:
    dummy = pd.get_dummies(df[i], drop_first=True, prefix = i)
    df = pd.concat([df,dummy], axis= 1)
    df.drop([i], inplace=True, axis= 1)

traint = df[:18832]
testt = df[18832:]
testt.drop(['breed_category'], axis=1, inplace=True)












np.random.seed(42)
y = train['breed_category']
X = train.drop(['breed_category'], axis=1)

# #Oversampling with SMOTETomek
# from imblearn.combine import SMOTETomek
# smt = SMOTETomek(sampling_strategy = 'minority', random_state=1996)
# X_res, y_res = smt.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, test_size=0.25, random_state=42)

sc_X = RobustScaler()
X2_train = sc_X.fit_transform(X_train)
X2_test = sc_X.fit_transform(X_test)
testt = sc_X.fit_transform(test)

# X_sm = sm.add_constant(X)
# model = sm.OLS(y,X_sm)
# print(model.fit().summary())




from sklearn.ensemble import ExtraTreesClassifier

ss2 = ExtraTreesClassifier()
print(cv_f1_w(ss2).mean())
ss2.fit(X2_train, y_train)
# l = pd.Series(ss.feature_importances_ * 100)
ss_pred = ss2.predict(X2_test)
print('ExtraTreesClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, ss_pred, average='weighted'))
print(classification_report(y_test,ss_pred))
print(confusion_matrix(y_test, ss_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, ss_pred),annot = True,cmap='BrBG')
plt.show()





print('\nwith non-scaled features:')
ada2 = AdaBoostClassifier()
print(cv_f1_w(ada2).mean())
ada2.fit(X2_train, y_train)
ada_pred = ada2.predict(X2_test)
print('AdaBoostClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, ada_pred, average='weighted'))
print(classification_report(y_test,ada_pred))
print(confusion_matrix(y_test, ada_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, ada_pred),annot = True,cmap='BrBG')
plt.show()




print('with non-scaled features:')
lm2 = LogisticRegression(random_state=42) #(penalty='l1', solver='liblinear',C=1)
print(cv_f1_w(lm2).mean())
lm2.fit(X2_train,y_train)
lm1_pred = lm2.predict(X2_test)
print('Logistic Regression Performance:')
print('f1_score:', metrics.f1_score(y_test, lm1_pred, average='weighted'))
print(classification_report(y_test,lm1_pred))
print(confusion_matrix(y_test, lm1_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, lm1_pred),annot = True,cmap='BrBG')
plt.show()

# param_grid = {'penalty' : ['l1', 'l2'],
#     'C' : 10.0**np.arange(-3,4),
#     'solver' : ['liblinear', 'lbfgs']}
# clf = GridSearchCV(estimator = LogisticRegression(), param_grid = param_grid, refit = True,
#                     cv = 3, verbose=True, n_jobs=-1, scoring = 'f1_weighted')
# clf.fit(X_train,y_train)
# print(clf.best_params_)
# print(clf.best_estimator_)




from sklearn.naive_bayes import GaussianNB
print('with non-scaled features:')
nb2 = GaussianNB()
print(cv_f1_w(nb2).mean())
nb2.fit(X2_train,y_train)
nb2_pred = nb2.predict(X2_test)
print('GaussianNB Performance:')
print('f1_score:', metrics.f1_score(y_test, nb2_pred, average='weighted'))
print(classification_report(y_test,nb2_pred))
print(confusion_matrix(y_test, nb2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, nb2_pred),annot = True,cmap='BrBG')
plt.show()




from sklearn.ensemble import BaggingClassifier
print('\nwith non-scaled features:')
bgc2 = BaggingClassifier()
print(cv_f1_w(bgc2).mean())
bgc2.fit(X2_train, y_train)
bgc2_pred = bgc2.predict(X2_test)
print('BaggingClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, bgc2_pred, average='weighted'))
print(classification_report(y_test,bgc2_pred))
print(confusion_matrix(y_test, bgc2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, bgc2_pred),annot = True,cmap='BrBG')
plt.show()




from sklearn.neural_network import MLPClassifier
print('with non-scaled features:')
mlp2 = MLPClassifier(random_state=42)
print(cv_f1_w(mlp2).mean())
mlp2.fit(X2_train,y_train)
mlp2_pred = mlp2.predict(X2_test)
print('MLPClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, mlp2_pred, average='weighted'))
print(classification_report(y_test,mlp2_pred))
print(confusion_matrix(y_test, mlp2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, mlp2_pred),annot = True,cmap='BrBG')
plt.show()



print('\nwith non-scaled features:')
DT2 = DecisionTreeClassifier()
print(cv_f1_w(DT2).mean())
DT2.fit(X2_train, y_train)
DT_pred = DT2.predict(X2_test)
print('DecisionTreeClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, DT_pred, average='weighted'))
print(classification_report(y_test,DT_pred))
print(confusion_matrix(y_test, DT_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, DT_pred),annot = True,cmap='BrBG')
plt.show()




print('with non-scaled features:')
rf2 = RandomForestClassifier(max_features='sqrt', n_estimators=300) #(max_features='sqrt', n_estimators=1500)
print(cv_f1_w(rf2).mean())
rf2.fit(X2_train,y_train)
rf2_pred = rf2.predict(X2_test)
print('RandomForestClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, rf2_pred, average='weighted'))
print(classification_report(y_test,rf2_pred))
print(confusion_matrix(y_test, rf2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, rf2_pred),annot = True,cmap='BrBG')
plt.show()

# rf_param_grid = {'max_features': ['sqrt', 'auto','log2'],
#               'min_samples_leaf': [1, 3, 5, 8],
#               'n_estimators': [100, 300, 500, 800, 1500]}
# rf_grid = GridSearchCV(estimator= RandomForestClassifier(), param_grid = rf_param_grid, refit = True
#                        scoring='f1_weighted' , n_jobs=-1, verbose=True, cv = 5)
# rf_grid.fit(X2_train,y_train)
# print(rf_grid.best_params_)
# print(rf_grid.best_estimator_)




print('with non-scaled features:')
svr2 = SVC(C=10, gamma=0.01, probability=False, kernel = 'rbf',
           decision_function_shape="ovr")
print(cv_f1_w(svr2).mean())
svr2.fit(X2_train,y_train)
svr_pred = svr2.predict(X2_test)
print('SVC Performance:')
print('f1_score:', metrics.f1_score(y_test, svr_pred, average='weighted'))
print(classification_report(y_test,svr_pred))
print(confusion_matrix(y_test, svr_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, svr_pred),annot = True,cmap='BrBG')
plt.show()

# param_grid = {'C': [1, 10, 25, 80, 100], 'gamma': [0.1, 0.01,0.001,0.0001]}#, 'kernel': ['linear', 'rbf']} 
# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3,cv=5,n_jobs=-1,scoring='f1_weighted')
# grid.fit(X2_train,y_train)
# print(grid.best_params_)
# print(grid.best_estimator_)





print('with non-scaled features:')
xbgc2 = xgb.XGBClassifier()
print(cv_f1_w(xbgc2).mean())
xbgc2.fit(X2_train,y_train)
xbgc2_pred = xbgc2.predict(X2_test)
print('XGB Performance:')
print('f1_score:', metrics.f1_score(y_test, xbgc2_pred, average='weighted'))
print(classification_report(y_test,xbgc2_pred))
print(confusion_matrix(y_test, xbgc2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, xbgc2_pred),annot = True,cmap='BrBG')
plt.show()

# params = {"learning_rate"    : [  0.1, 0.01,  0.001],
#   "max_depth"        : [  3, 5, 8],
#   "min_child_weight" : [  3, 5, 8 ],
#   "colsample_bytree" : [ 0.1, 0.3, 0.5],
#   'n_estimators': [ 100, 300, 500, 800, 1500]}
# xgb_tuning = GridSearchCV(estimator = xgb.XGBClassifier(random_state=1996),
#                           param_grid = params, refit = True
#                           n_jobs=-1, scoring = 'f1_weighted',
#                           cv=5, verbose=True)
# xgb_tuning.fit(X2_train,y_train)
# print(xgb_tuning.best_params_)
# print(xgb_tuning.best_estimator_)




print('\nwith non-scaled features:')
gbm2 = GradientBoostingClassifier(random_state=42)
print(cv_f1_w(gbm2).mean())
gbm2.fit(X2_train, y_train)
gbm2_pred = gbm2.predict(X2_test)
print('Gradiant Boosting Performance:')
print('f1_score:', metrics.f1_score(y_test, gbm2_pred, average='weighted'))
print(classification_report(y_test,gbm2_pred))
print(confusion_matrix(y_test, gbm2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, gbm2_pred),annot = True,cmap='BrBG')
plt.show()

# gbm_param_grid = {'learning_rate':[0.1, 0.01, 0.001], 
#             'n_estimators':[100, 300, 500, 1000],
#           'max_depth':[3, 5, 8],
#           'min_samples_split':[10, 20],
#           'max_features':[4, 7]}
# gbm_tuning = GridSearchCV(estimator =GradientBoostingClassifier(random_state=1996),
#                           param_grid = gbm_param_grid, verbose = True,
#                           n_jobs=-1, scoring='f1_weighted'
#                           cv=5, refit = True)
# gbm_tuning.fit(X_train,y_train)
# print(gbm_tuning.best_params_)
# print(gbm_tuning.best_estimator_)




print('\nwith non-scaled features:')
knn = KNeighborsClassifier(n_neighbors=18)
print(cv_f1_w(knn).mean())
knn.fit(X2_train, y_train)
knn_pred = knn.predict(X2_test)
print('KNeighborsClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, knn_pred, average='weighted'))
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test, knn_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, knn_pred),annot = True,cmap='BrBG')
plt.show()

E = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    E.append(metrics.f1_score(y_test, knn_preds, average='weighted'))

fig = plt.figure(figsize=(10,8))    
plt.plot(range(1,51), E, ls = '--',lw = 2 , markersize = 4, marker = 'o', markerfacecolor = 'red',color = 'blue')





print('\nwith non-scaled features:')
lgbc2 = lgb.LGBMClassifier()
print(cv_f1_w(lgbc2).mean())
lgbc2.fit(X2_train, y_train)
lgbc2_pred = lgbc2.predict(X2_test)
print('LGBMClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, lgbc2_pred, average='weighted'))
print(classification_report(y_test,lgbc2_pred))
print(confusion_matrix(y_test, lgbc2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, lgbc2_pred),annot = True,cmap='BrBG')
plt.show()


print('with non-scaled features:')
cbr2 = cb.CatBoostClassifier(random_state=42)
print(cv_f1_w(cbr2).mean())
cbr2.fit(X2_train, y_train)
cbr2_pred = cbr2.predict(X2_test)
print('CatBoostClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, cbr2_pred, average='weighted'))
print(classification_report(y_test,cbr2_pred))
print(confusion_matrix(y_test, cbr2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, cbr2_pred),annot = True,cmap='BrBG')
plt.show()



# ('rf2',rf2),('lgbc2',lgbc2),('xbgc2',xbgc2),('svr2',svr2),
# ('bgc2',bgc2),('mlp2',mlp2),('ss2',ss2),('bgc2',bgc2),
#                           ('svr2',svr2)
#                           ('DT2',DT2),('rf2',rf2),
#                           ('xbgc2',xbgc2),('lgbc2',lgbc2)
# ('ada2',ada2),('svr2',svr2)

vclf2 = VotingClassifier([('bgc2',bgc2),('svr2',svr2),('ada2',ada2),('ss2',ss2),
                         ('gbm2',gbm2),('rf2',rf2)], 
                          voting='hard')
print(cv_f1_w(vclf2).mean())
vclf2.fit(X2_train, y_train)
vclf2_pred = vclf2.predict(X2_test)
# prob = vclf2.predict_proba(X2_test)
print('VotingClassifier Performance:')
print('f1_score:', metrics.f1_score(y_test, vclf2_pred, average = 'weighted'))
print('accuracy_score:', metrics.accuracy_score(y_test, vclf2_pred))
print(classification_report(y_test,vclf2_pred))
print(confusion_matrix(y_test, vclf2_pred))
fig = plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, vclf2_pred),annot = True,cmap='BrBG')
plt.show()



sub = pd.DataFrame()
sssss = pd.read_csv('HE/Pet_adoption/test.csv')
sub['pet_id'] = sssss['pet_id']
sub['breed_category'] = vclf2.predict(tests)
sub['pet_category'] = vclf.predict(tests)

sub.to_csv('HE/Pet_adoption/Submission.csv', index = False)




import pickle 
# save the model to disk
# pickle.dump(vclf2, open('vclf2_anova2.pkl', 'wb'))
# pickle.dump(vclf, open('vclf_anova2.pkl', 'wb'))
 
 
# load the model from disk
loaded_model = pickle.load(open('vclf_anova.pkl', 'rb'))
loaded_model2 = pickle.load(open('vclf2_anova.pkl', 'rb'))

# Predictions
sub = pd.DataFrame()
sssss = pd.read_csv('HE/Pet_adoption/test.csv')
sub['pet_id'] = sssss['pet_id']
sub['breed_category'] = loaded_model2.predict(test2)
sub['pet_category'] = loaded_model.predict(test1)

# 91.01855 Anova with 20 Kbest features
# 90.99165 Anova with 20 Kbest features for model2
sub.to_csv('HE/Pet_adoption/Submission.csv', index = False)

