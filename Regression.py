import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import boxcox
from scipy import stats
import pylab
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
sns.set(style='ticks')


train = pd.read_csv('Dengue/dengue_features_train.csv')
test = pd.read_csv('Dengue/dengue_features_test.csv')
trainy = pd.read_csv('Dengue/dengue_labels_train.csv')
sub = pd.read_csv('Dengue/submission_format.csv')
train = pd.concat([train,trainy['total_cases']], axis= 1)

desc = train.describe().T
# print(train.info())
# print(train.columns)


na1 = train.isnull().sum()*100/1456
na1 = na1[na1 > 0]
# plt.figure(figsize = (15,10))
# sns.heatmap(train.isnull(),cmap='gray')

# median1 = []
# for i in na1.index: 
#     median1.append(train[i].median())
# f = 0
# for i in na1.index: 
#     train[i] = train[i].fillna(median1[f])
#     f += 1


na2 = test.isnull().sum()*100/1456 
na2 = na2[na2 > 0]
# plt.figure(figsize = (15,10))
# sns.heatmap(test.isnull(),cmap='gray')

# median2 = []
# for i in na2.index: 
#     median2.append(test[i].median())
# f = 0
# for i in na2.index: 
#     test[i] = test[i].fillna(median2[f])
#     f += 1


train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)


# plt.figure(figsize = (15,10))
# sns.pairplot(train, hue = 'city')



train['week_start_date_Year'] = train['week_start_date'].apply(lambda x: int(x[:4]))
train['week_start_date_Month'] = pd.to_datetime(train['week_start_date']).dt.month
train['week_start_date_Day'] = pd.to_datetime(train['week_start_date']).dt.day
train['week_start_date_Dayofweek'] = pd.to_datetime(train['week_start_date']).dt.dayofweek
train['week_start_date_DayOfyear'] = pd.to_datetime(train['week_start_date']).dt.dayofyear
train['week_start_date_Week_No'] = pd.to_datetime(train['week_start_date']).dt.week

for i in train[(train['week_start_date_Day'] == 1) & 
               (train['week_start_date_Month'] == 1)].index:
    train.loc[i, 'week_start_date_Week_No'] = 1
  
train['week_start_date_Quarter'] = pd.to_datetime(train['week_start_date']).dt.quarter 
train['week_start_date_Is_month_start'] = pd.to_datetime(train['week_start_date']).dt.is_month_start
train['week_start_date_Is_month_end'] = pd.to_datetime(train['week_start_date']).dt.is_month_end
train['week_start_date_Is_quarter_start'] = pd.to_datetime(train['week_start_date']).dt.is_quarter_start
train['week_start_date_Is_quarter_end'] = pd.to_datetime(train['week_start_date']).dt.is_quarter_end
train['week_start_date_Is_year_start'] = pd.to_datetime(train['week_start_date']).dt.is_year_start
# train['week_start_date_Is_year_end'] = pd.to_datetime(train['week_start_date']).dt.is_year_end
train['week_start_date_Is_weekend'] = np.where(train['week_start_date_Dayofweek'].isin([5,6]),1,0)
train['week_start_date_Is_weekday'] = np.where(train['week_start_date_Dayofweek'].isin([0,1,2,3,4]),1,0)

train['week_start_date_Is_weekend'] = train['week_start_date_Is_weekend'].apply( lambda x: bool(x))
train['week_start_date_Is_weekday'] = train['week_start_date_Is_weekday'].apply( lambda x: bool(x))



test['week_start_date_Year'] = test['week_start_date'].apply(lambda x: int(x[:4]))
test['week_start_date_Month'] = pd.to_datetime(test['week_start_date']).dt.month
test['week_start_date_Day'] = pd.to_datetime(test['week_start_date']).dt.day
test['week_start_date_Dayofweek'] = pd.to_datetime(test['week_start_date']).dt.dayofweek
test['week_start_date_DayOfyear'] = pd.to_datetime(test['week_start_date']).dt.dayofyear
test['week_start_date_Week_No'] = pd.to_datetime(test['week_start_date']).dt.week

for i in test[(test['week_start_date_Day'] == 1) & 
               (test['week_start_date_Month'] == 1)].index:
    test.loc[i, 'week_start_date_Week_No'] = 1

test['week_start_date_Quarter'] = pd.to_datetime(test['week_start_date']).dt.quarter 
test['week_start_date_Is_month_start'] = pd.to_datetime(test['week_start_date']).dt.is_month_start
test['week_start_date_Is_month_end'] = pd.to_datetime(test['week_start_date']).dt.is_month_end
test['week_start_date_Is_quarter_start'] = pd.to_datetime(test['week_start_date']).dt.is_quarter_start
test['week_start_date_Is_quarter_end'] = pd.to_datetime(test['week_start_date']).dt.is_quarter_end
test['week_start_date_Is_year_start'] = pd.to_datetime(test['week_start_date']).dt.is_year_start
# test['week_start_date_Is_year_end'] = pd.to_datetime(test['week_start_date']).dt.is_year_end
test['week_start_date_Is_weekend'] = np.where(test['week_start_date_Dayofweek'].isin([5,6]),1,0)
test['week_start_date_Is_weekday'] = np.where(test['week_start_date_Dayofweek'].isin([0,1,2,3,4]),1,0)

test['week_start_date_Is_weekend'] = test['week_start_date_Is_weekend'].apply( lambda x: bool(x))
test['week_start_date_Is_weekday'] = test['week_start_date_Is_weekday'].apply( lambda x: bool(x))


train.drop(['week_start_date','year', 'weekofyear'], axis = 1, inplace=True)
test.drop(['week_start_date','year', 'weekofyear'], axis = 1, inplace=True)



def calc_year(x):
    if x <= 1995:
        return 'A'
    elif x <= 2000:
        return 'B'
    elif x <= 2005:
        return 'C'
    elif x <= 2015:
        return 'D'

train['year_group'] = pd.Series()
test['year_group'] = pd.Series()
train['year_group'] = train['week_start_date_Year'].apply(lambda x: calc_year(x))
test['year_group'] = test['week_start_date_Year'].apply(lambda x: calc_year(x))


def calc_day(x):
    if x <= 10:
        return 'P'
    elif x <= 20:
        return 'Q'
    else:
        return 'R'

train['day_group'] = pd.Series()
test['day_group'] = pd.Series()
train['day_group'] = train['week_start_date_Day'].apply(lambda x: calc_day(x))
test['day_group'] = test['week_start_date_Day'].apply(lambda x: calc_day(x))


def calc_DayOfYear(x):
    if x <= 71:
        return 'V'
    elif x <= 142:
        return 'W'
    elif x <= 213:
        return 'X'
    elif x <= 284:
        return 'Y'
    else:
        return 'Z'

train['DayOfYear_group'] = pd.Series()
test['DayOfYear_group'] = pd.Series()
train['DayOfYear_group'] = train['week_start_date_DayOfyear'].apply(lambda x: calc_DayOfYear(x))
test['DayOfYear_group'] = test['week_start_date_DayOfyear'].apply(lambda x: calc_DayOfYear(x))


def calc_Week_No(x):
    if x <= 10:
        return '1st'
    elif x <= 20:
        return '2nd'
    elif x <= 30:
        return '3rd'
    elif x <= 40:
        return '4th'
    else:
        return '5th'

train['Week_No_group'] = pd.Series()
test['Week_No_group'] = pd.Series()
train['Week_No_group'] = train['week_start_date_Week_No'].apply(lambda x: calc_Week_No(x))
test['Week_No_group'] = test['week_start_date_Week_No'].apply(lambda x: calc_Week_No(x))



num = []
cat_bool = []

for i in train.columns:
    if (train[i].dtypes != 'object') & (train[i].dtypes != 'bool'):
        num.append(i)
    else :
        cat_bool.append(i)
        
        
R = ['week_start_date_Quarter','week_start_date_Month','week_start_date_Dayofweek']
# A = ['week_start_date_Quarter','week_start_date_Month','week_start_date_Dayofweek',
#      'year_group','Week_No_group','day_group','DayOfYear_group']
for i in R:
    num.remove(i)
    cat_bool.append(i)




# for i in num[:3]:
#     plt.figure(figsize =(6,4)) 
#     ax = sns.distplot(train[i], hist=True, color="red", kde = True,
#                       kde_kws={"shade": True}, hist_kws = dict(edgecolor="k", linewidth=3))

#     plt.tight_layout()
# for i in num:
#     plt.figure(figsize=(8,6))
#     stats.probplot(train[i], plot=pylab, dist="norm")
#     pylab.show()
#     print(i,train[i].skew(),train[i].kurt())

# dfdffddf

for x in num:
    # train[train[i]>up_lim][i].count()
    Q1 = np.percentile(train['total_cases'], 25)
    Q3 = np.percentile(train['total_cases'], 75)
    IQR = Q3 - Q1
    low_lim = Q1 - 1.7 * IQR 
    up_lim = Q3 + 1.7 * IQR 
    train = train[train['total_cases'] <= up_lim]
    train = train[train['total_cases'] >= low_lim]
# train[i] = train[i].apply(lambda x: iqr(x))



# sasddsa

# corr = train.corr()['total_cases']
# for i in range(len(corr)):
#     if corr[i] < 0:
#         corr[i] *= -1   
# corr_top10 = list(corr.sort_values()[-12:-2].index)
    
# plt.figure(figsize = (15,10))
# sns.heatmap(train.corr(), annot = True)

num.remove('total_cases')



# here, tried with top 15 correlated feats, but the results were bad

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method = 'yeo-johnson', standardize = True, copy = False)
train[num] = pt.fit_transform(train[num])
test[num] = pt.fit_transform(test[num])

# adsda

# for i in num:
#     if i == 'total_cases':
#         train[i] = (train[i])**2
#     else :
#         train[i] = (train[i])**2
#         test[i] = (test[i])**2

# fafaffsa

# for i in num:
#     plt.figure(figsize =(6,4)) 
#     ax = sns.boxplot(train[i],palette ='Set3')

# for i in num:
#     train[i] = train[(train[i] > int(-3)) & (train[i] < int(3))][i]
#     # train[i] = train[train[i] < int(3)][i]
#     test[i] = test[(test[i] > int(-3)) & (test[i] < int(3))][i]
#     # test[i] = test[test[i] < int(3)][i]



# np.random.seed(42)
# y = train['total_cases']
# X = train.drop(['total_cases'], axis=1)

# from sklearn.feature_selection import SelectKBest, f_classif
# X_new = SelectKBest(f_classif, k=20).fit_transform(X, y)
# adsasdasdasd



ww = []
for i in cat_bool:
    ww.append(train[i].value_counts().count())


df = pd.concat([train,test], axis= 0)
for i in cat_bool:
    dummy = pd.get_dummies(df[i], drop_first=True, prefix = i)
    df = pd.concat([df,dummy], axis= 1)
    df.drop([i], inplace=True, axis= 1)

train = df[:1272]
test = df[1272:]#1348
test.drop(['total_cases'], axis=1, inplace=True)





# dadasad
# X_train[X_train.isin([np.nan, np.inf, -np.inf]).any(1)]

skf = KFold(n_splits=10, random_state=47, shuffle=True)
def cv_mae(model):
    maes = -cross_val_score(model, X_train, y_train.values.ravel(), 
                           scoring='neg_mean_squared_error', 
                           cv=skf, n_jobs=-1, verbose = True)
    return (maes)




from sklearn.decomposition import PCA

scaled_data = train.iloc[:,:25]
scaled_data.drop(['total_cases'], axis=1)
pca = PCA(n_components=10)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
aa = pd.DataFrame(x_pca)


train_pca = train.iloc[:,25:]
ss = train['total_cases']

aa.reset_index(drop=True, inplace=True)
train_pca.reset_index(drop=True, inplace=True)
ss.reset_index(drop=True, inplace=True)

df1 = pd.concat([aa, train_pca, ss], axis= 1, ignore_index=True)



scaled_data = test.iloc[:,:24]
# scaled_data.drop(['total_cases'], axis=1)
pca = PCA(n_components=10)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
aa = pd.DataFrame(x_pca)


test_pca = test.iloc[:,24:]
# ss = train['total_cases']

aa.reset_index(drop=True, inplace=True)
train_pca.reset_index(drop=True, inplace=True)
# ss.reset_index(drop=True, inplace=True)

df2 = pd.concat([aa, test_pca], axis= 1, ignore_index=True)







np.random.seed(47)
y = df1.iloc[:,51:]
X = df1.iloc[:,:51]

# train['city'].value_counts()
# sns.barplot(x = 'city_sj', y = 'total_cases',data = train, palette="viridis")

# stratifying on city but nothing much of helps bcz of same ratio inn the train ands test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=47)


# sc_X = StandardScaler()
# X2_train = sc_X.fit_transform(X_train)
# X2_test = sc_X.fit_transform(X_test)
# tests = sc_X.fit_transform(test)

# X_sm = sm.add_constant(X)
# model = sm.OLS(y,X_sm)
# print(model.fit().summary())




lm1 = LinearRegression()
print(cv_mae(lm1).mean())
lm1.fit(X_train,y_train)
lm1_pred = lm1.predict(X_test)
print('Linear Regression Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, lm1_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm1_pred)))
print('R2_Score: ', metrics.r2_score(y_test, lm1_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,lm1_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('LinearRegression Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, lm1_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, lm1_pred))
R2_Score = metrics.r2_score(y_test, lm1_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()
# print('Estimated coefficients for the linear regression: ',lm1.coef_)
# print('Independent term: ', lm1.intercept_)




ls = Lasso()
print(cv_mae(ls).mean())
ls.fit(X_train,y_train)
ls_pred = ls.predict(X_test)
print('Lasso Regression Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, ls_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ls_pred)))
print('R2_Score: ', metrics.r2_score(y_test, ls_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,ls_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('Lasso Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, ls_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, ls_pred))
R2_Score = metrics.r2_score(y_test, ls_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



rg = Ridge()
print(cv_mae(rg).mean())
rg.fit(X_train,y_train)
rg_pred = rg.predict(X_test)
print('Ridge Regression Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, rg_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rg_pred)))
print('R2_Score: ', metrics.r2_score(y_test, rg_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,rg_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('Ridge Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, rg_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, rg_pred))
R2_Score = metrics.r2_score(y_test, rg_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



rf = RandomForestRegressor(criterion = 'mae',random_state=31)
print(cv_mae(rf).mean())
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print('RandomForestRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, rf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
print('R2_Score: ', metrics.r2_score(y_test, rf_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,rf_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('RFR Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, rf_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, rf_pred))
R2_Score = metrics.r2_score(y_test, rf_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()
feat_imp = pd.Series(rf.feature_importances_, list(X_train)).sort_values(ascending=False)
fig = plt.figure(figsize=(12, 6))
feat_imp.plot(kind='bar', title='Importance of Features',color = 'coral')
plt.ylabel('Feature Importance Score')
plt.grid()
plt.show()


# rf_param_grid = {'max_features': ['sqrt', 'auto','log2'],
#               'min_samples_leaf': [1, 3, 5, 8],
#               'min_samples_split': [2, 3, 5, 8],
#               'n_estimators': [100, 300, 500, 800, 1500]}
# rf_grid = GridSearchCV(estimator= RandomForestRegressor(criterion = 'mae',random_state=31),
#                         param_grid = rf_param_grid, refit = True,
#                         scoring='neg_mean_absolute_error', 
#                         n_jobs=-1, verbose=True, cv = 3)
# rf_grid.fit(X_train,y_train)
# print(rf_grid.best_params_)
# print(rf_grid.best_estimator_)

# asfdsggs


gbm1 = GradientBoostingRegressor()
print(cv_mae(gbm1).mean())
gbm1.fit(X_train, y_train)
gbm1_pred = gbm1.predict(X_test)
print('Gradiant Boosting Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, gbm1_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, gbm1_pred)))
print('R2_Score: ', metrics.r2_score(y_test, gbm1_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,gbm1_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('GBM Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, gbm1_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, gbm1_pred))
R2_Score = metrics.r2_score(y_test, gbm1_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()
feat_imp = pd.Series(gbm1.feature_importances_, list(X_train)).sort_values(ascending=False)
fig = plt.figure(figsize=(8, 5))
feat_imp.plot(kind='bar', title='Importance of Features', color ='coral')
plt.ylabel('Feature Importance Score')
plt.grid()
plt.show()

# gbm_param_grid = {'learning_rate':[0.1, 0.01, 0.001], 
#             'n_estimators':[100, 300, 500, 800, 1500],
#           'max_depth':[3, 5, 8],
#           'min_samples_split':[10, 20],
#           'max_features':[4, 7]}
# gbm_tuning = GridSearchCV(estimator =GradientBoostingRegressor(random_state=31),
#                           param_grid = gbm_param_grid, verbose = True,
#                           n_jobs=-1, scoring='neg_mean_absolute_error',
#                           cv=3, refit = True)
# gbm_tuning.fit(X_train,y_train)
# print(gbm_tuning.best_params_)
# print(gbm_tuning.best_estimator_)



svr = SVR()
print(cv_mae(svr).mean())
svr.fit(X_train,y_train)
svr_pred = svr.predict(X_test)
print('SVR Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
print('R2_Score: ', metrics.r2_score(y_test, svr_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,svr_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('SVR Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, svr_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, svr_pred))
R2_Score = metrics.r2_score(y_test, svr_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()


# param_grid = {'C': [1, 10, 25, 80, 100], 'gamma': [0.1, 0.01,0.001,0.0001], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']} 
# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=True,cv=3,n_jobs=-1,scoring='neg_mean_absolute_error')
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# print(grid.best_estimator_)



lsvr = LinearSVR()
print(cv_mae(lsvr).mean())
lsvr.fit(X_train,y_train)
lsvr_pred = lsvr.predict(X_test)
print('LinearSVR Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, lsvr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lsvr_pred)))
print('R2_Score: ', metrics.r2_score(y_test, lsvr_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,lsvr_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('LinearSVR Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, lsvr_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, lsvr_pred))
R2_Score = metrics.r2_score(y_test, lsvr_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



# E = []
# for i in range(1,101):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     knn_preds = knn.predict(X_test)
#     E.append(metrics.mean_absolute_error(y_test, knn_preds))

# fig = plt.figure(figsize=(10,8))    
# plt.plot(range(1,101), E, ls = '--',lw = 2 , markersize = 4, marker = 'o', markerfacecolor = 'red',color = 'blue')
# plt.grid()



knnr = KNeighborsRegressor(n_neighbors=2)
print(cv_mae(knnr).mean())
knnr.fit(X_train,y_train)
knnr_pred = knnr.predict(X_test)
print('KNeighborsRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, knnr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knnr_pred)))
print('R2_Score: ', metrics.r2_score(y_test, knnr_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,knnr_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('KNeighborsRegressor Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, knnr_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, knnr_pred))
R2_Score = metrics.r2_score(y_test, knnr_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



dt = DecisionTreeRegressor()
print(cv_mae(dt).mean())
dt.fit(X_train,y_train)
dt_pred = dt.predict(X_test)
print('DecisionTreeRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, dt_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dt_pred)))
print('R2_Score: ', metrics.r2_score(y_test, dt_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,dt_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('DecisionTreeRegressor Prediction Performance ') 
MAE = metrics.mean_absolute_error(y_test, dt_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, dt_pred))
R2_Score = metrics.r2_score(y_test, dt_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



xbgr = xgb.XGBRegressor()
print(cv_mae(xbgr).mean())
xbgr.fit(X_train,y_train)
xbgr_pred = xbgr.predict(X_test)
print('XGB Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, xbgr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xbgr_pred)))
print('R2_Score: ', metrics.r2_score(y_test, xbgr_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,xbgr_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('XGB Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, xbgr_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, xbgr_pred))
R2_Score = metrics.r2_score(y_test, xbgr_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()
feat_imp = pd.Series(xbgr.feature_importances_, list(X_train)).sort_values(ascending=False)
fig = plt.figure(figsize=(8, 5))
feat_imp.plot(kind='bar', title='Importance of Features', color ='coral')
plt.ylabel('Feature Importance Score')
plt.grid()
plt.show()

# params = {"learning_rate"    : [  1, 0.1, 0.01,  0.001],
#   "max_depth"        : [  3, 5, 8],
#   "min_child_weight" : [  3, 5, 8 ],
#   "colsample_bytree" : [ 0.1, 0.3, 0.5],
#   'n_estimators': [ 100, 300, 500, 800, 1500]}
# xgb_tuning = GridSearchCV(estimator = xgb.XGBClassifier(random_state=31),
#                           param_grid = params, refit = True,
#                           n_jobs=-1, scoring = 'neg_median_absolute_error',
#                           cv=3, verbose=True)
# xgb_tuning.fit(X_train,y_train)
# print(xgb_tuning.best_params_)
# print(xgb_tuning.best_estimator_)



lgbc = lgb.LGBMRegressor()
print(cv_mae(lgbc).mean())
lgbc.fit(X_train, y_train)
lgbc_pred = lgbc.predict(X_test)
print('LGBMRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, lgbc_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgbc_pred)))
print('R2_Score: ', metrics.r2_score(y_test, lgbc_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,lgbc_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('LGBMRegressor Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, lgbc_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, lgbc_pred))
R2_Score = metrics.r2_score(y_test, lgbc_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()

 

ada = AdaBoostRegressor()
print(cv_mae(ada).mean())
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
print('AdaBoostRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, ada_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ada_pred)))
print('R2_Score: ', metrics.r2_score(y_test, ada_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,ada_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('AdaBoostRegressor Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, ada_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, ada_pred))
R2_Score = metrics.r2_score(y_test, ada_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



bgc = BaggingRegressor()
print(cv_mae(bgc).mean())
bgc.fit(X_train, y_train)
bgc_pred = bgc.predict(X_test)
print('BaggingRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, bgc_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, bgc_pred)))
print('R2_Score: ', metrics.r2_score(y_test, bgc_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,bgc_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('BaggingRegressor Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, bgc_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, bgc_pred))
R2_Score = metrics.r2_score(y_test, bgc_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



ss = ExtraTreesRegressor(criterion = 'mae', random_state=47)
ss_score = cv_mae(ss)
print(ss_score.mean())
ss.fit(X_train, y_train)
ss_pred = ss.predict(X_test)
print('ExtraTreesRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, ss_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ss_pred)))
print('R2_Score: ', metrics.r2_score(y_test, ss_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,ss_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('ExtraTreesRegressor Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, ss_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, ss_pred))
R2_Score = metrics.r2_score(y_test, ss_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()
# 53.84788415730336
# ExtraTreesRegressor Performance:
# MAE: 4.0803125
# RMSE: 5.625908220329534
# R2_Score:  0.8547231819000127
# import pickle 
# # save the model to disk
# pickle.dump(ss, open('ss.pkl', 'wb'))

# # load the model from disk
# loaded_model = pickle.load(open('ss.pkl', 'rb'))

asfafasfsafas

cbr = cb.CatBoostRegressor()
print(cv_mae(cbr).mean())
cbr.fit(X_train, y_train)
cbr_pred = cbr.predict(X_test)
print('CatBoostRegressor Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, cbr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, cbr_pred)))
print('R2_Score: ', metrics.r2_score(y_test, cbr_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,cbr_pred,color='g')
plt.xlabel('total_case') 
plt.ylabel('Predictions') 
plt.title('CatBoostRegressor Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, cbr_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, cbr_pred))
R2_Score = metrics.r2_score(y_test, cbr_pred)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()



import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# sc_X = StandardScaler()
# X2_train = sc_X.fit_transform(X_train)
# X2_test = sc_X.fit_transform(X_test)
# test2 = sc_X.fit_transform(test)

model = Sequential()

model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1028, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'relu'))

model.compile(loss = 'mae', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience=20)

model.fit(x=X_train, y = y_train, epochs=1000, validation_data=(X_test,y_test), callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)

fig = plt.figure(figsize=(10,8))
losses.plot()

predictions = model.predict(X_test)

print('NN Performance:')
print('MAE:', metrics.mean_absolute_error(np.array(y_test), predictions))
print('MSE:', metrics.mean_squared_error(np.array(y_test), predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(np.array(y_test), predictions)))
print('R2_Score: ', metrics.r2_score(np.array(y_test), predictions))
fig = plt.figure(figsize=(8, 5))
sns.regplot(np.array(y_test), predictions,color='g',scatter_kws={'alpha':0.5})
plt.xlabel('Sale Price') 
plt.ylabel('Prediction') 
plt.title('NN Prediction Performance ')  
MAE = metrics.mean_absolute_error(y_test, predictions)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
R2_Score = metrics.r2_score(y_test, predictions)
plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
plt.grid()
plt.show()

# model.save('Dengue_NN.h5')
# NN Performance:
# MAE: 6.19209872451547
# MSE: 119.95812351895029
# RMSE: 10.952539592211036
# R2_Score:  0.9285493855542055



# from sklearn.ensemble import StackingRegressor
# estimators = [('rf',rf), ('bgc',bgc)]
# reg = StackingRegressor(estimators=estimators, 
#                         final_estimator=ExtraTreesRegressor())
# print(cv_mae(reg).mean())
# reg.fit(X_train,y_train)
# reg_pred = reg.predict(X_test)
# print('StackingRegressor Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, reg_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, reg_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, reg_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,reg_pred,color='g')
# plt.xlabel('total_case') 
# plt.ylabel('Predictions') 
# plt.title('StackingRegressor Prediction Performance ') 
# MAE = metrics.mean_absolute_error(y_test, reg_pred)
# RMSE = np.sqrt(metrics.mean_squared_error(y_test, reg_pred))
# R2_Score = metrics.r2_score(y_test, reg_pred)
# plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
# plt.grid()
# plt.show()



# X_trans = ss.predict(test)
# if X >= 0 and lambda_ == 0:
#     X = exp(X_trans) - 1
# elif X >= 0 and lambda_ != 0:
#     X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
# elif X < 0 and lambda_ != 2:
#     X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
# elif X < 0 and lambda_ == 2:
#     X = 1 - exp(-X_trans)

aa = np.floor(ss.predict(test))
pred_cases = pd.Series(aa.reshape(-1), dtype = 'int64')
sub['total_cases'] = pred_cases
sub.to_csv('Dengue/Submission.csv', index = False)



# ss 35.2830
# stacking 38.6226
# NN 30.7764



MAE = []
R2_Score = []
# np.random.seed(i)
y = train['total_cases']
X = train.drop(['total_cases'], axis=1)
for i in range(0,100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify = X['city_sj'] ,
                                                        random_state=i)
    
    ss = ExtraTreesRegressor(criterion = 'mae')
    # print(cv_mae(ss).mean())
    ss.fit(X_train, y_train)
    ss_pred = ss.predict(X_test)
    # print('ExtraTreesRegressor Performance:')
    # print('MAE:', metrics.mean_absolute_error(y_test, ss_pred))
    # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ss_pred)))
    # print('R2_Score: ', metrics.r2_score(y_test, ss_pred))
    # fig = plt.figure(figsize=(8, 5))
    # sns.regplot(y_test,ss_pred,color='g')
    # plt.xlabel('total_case') 
    # plt.ylabel('Predictions') 
    # plt.title('ExtraTreesRegressor Prediction Performance ')  
    MAE.append(metrics.mean_absolute_error(y_test, ss_pred))
    # RMSE = np.sqrt(metrics.mean_squared_error(y_test, ss_pred))
    R2_Score.append(metrics.r2_score(y_test, ss_pred))
    # plt.legend(title='Model', loc='lower right', labels=[MAE,RMSE,R2_Score])
    # plt.grid()
    # plt.show()




for i in num[:3]:
    plt.figure(figsize=[10,8])
    sns.scatterplot(x=train[i], y=train['total_cases'], hue = train['city'],palette ='viridis')


for i in num[:3]:
    sns.jointplot(x=train[i], y=train['total_cases'], kind="hex", color='#4CB391', space = 0.01) 
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.jointplot(x=train[i], y=train['total_cases'], kind="kde", color = 'purple',n_levels=5, cmap=cmap, shade = True)


for i in cat_bool[1:3]:
    plt.figure(figsize = (12,8))
    ax = sns.barplot(x = i,y = 'total_cases',data = train, palette="BrBG",hue = 'city')
    plt.tight_layout()
    plt.xticks( rotation=90)
    plt.show()

#Dropping it
train['week_start_date_Is_year_end'].value_counts()



rf_param_grid = {'max_features': ['sqrt', 'auto','log2'],
              'min_samples_leaf': [1, 3, 5, 8],
              'min_samples_split': [2, 3, 5, 8],
              'n_estimators': [100, 300, 500, 800, 1500]}
rf_grid = GridSearchCV(estimator= RandomForestRegressor(criterion = 'mae',random_state=31),
                        param_grid = rf_param_grid, refit = True,
                        scoring='neg_mean_absolute_error', 
                        n_jobs=-1, verbose=True)
rf_grid.fit(X_train,y_train)
print(rf_grid.best_params_)
print(rf_grid.best_estimator_)




params = {"learning_rate"    : [  1, 0.1, 0.01,  0.001],
  "max_depth"        : [  3, 5, 8],
  "min_child_weight" : [  3, 5, 8 ],
  "colsample_bytree" : [ 0.1, 0.3, 0.5],
  'n_estimators': [ 100, 300, 500, 800, 1500]}
xgb_tuning = GridSearchCV(estimator = xgb.XGBRegressor(random_state=31),
                          param_grid = params, refit = True,
                          n_jobs=-1, scoring = 'neg_median_absolute_error',
                          verbose=True)
xgb_tuning.fit(X_train,y_train)
print(xgb_tuning.best_params_)
print(xgb_tuning.best_estimator_)




rf_param_grid = {'max_features': ['sqrt', 'auto','log2'],
              'min_samples_leaf': [ 3, 5, 8],
              'min_samples_split': [ 3, 5, 8],
              'n_estimators': [100, 300, 500, 800, 1500]}
rf_grid = GridSearchCV(estimator= ExtraTreesRegressor(criterion = 'mae',random_state=31),
                        param_grid = rf_param_grid, refit = True,
                        scoring='neg_mean_absolute_error', 
                        n_jobs=-1, verbose=True)
rf_grid.fit(X_train,y_train)
print(rf_grid.best_params_)
print(rf_grid.best_estimator_)




