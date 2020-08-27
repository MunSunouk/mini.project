import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline  

youtube = pd.read_csv("/kaggle/input/youtube-new/INvideos.csv")
youtube = youtube.dropna(how='any',axis=0)
youtube.drop(['video_id','thumbnail_link'],axis=1,inplace=True)
youtube["len_title"]=title_len
publish_time = pd.to_datetime(youtube['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
youtube['publish_time'] = publish_time.dt.time
youtube['publish_date'] = publish_time.dt.date
youtube['publish_date'] = publish_time.dt.date
##youtube['trending_date'] = pd.to_datetime(USvideos['trending_date'], format='%y.%d.%m').dt.date
youtube['publish_hour'] = publish_time.dt.hour

youtube.drop(['trending_date','publish_date','publish_time','tags','title','description','channel_title'],axis=1,inplace=True)

views=youtube['views']
youtube_view=youtube.drop(['views'],axis=1,inplace=False)
train,test,y_train,y_test=train_test_split(youtube_view,views, test_size=0.2,shuffle=False)

#Lasso
youtube_view2 = youtube[['likes','dislikes','comment_count']]
train,test,y_train,y_test=train_test_split(youtube_view2,views, test_size=0.2,shuffle=False)
# lasso
model_lasso = Lasso().fit(train, y_train)

print("훈련 세트 점수: {:.2f}".format(model_lasso.score(train, y_train)))
print("테스트 세트 점수: {:.2f}".format(model_lasso.score(test, y_test)))
print("사용한 특성의 수: {}".format(   np.sum( model_lasso.coef_ != 0 )   ))

#Ridge

train,test,y_train,y_test=train_test_split(youtube_view,views, test_size=0.2,shuffle=False)
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(train, y_train)
print("훈련 세트 점수: {:.2f}".format(model.score(train, y_train)))
print("테스트 세트 점수: {:.2f}".format(model.score(test, y_test)))
print("사용한 특성의 수: {}".format(   np.sum( model.coef_ != 0 )   ))

# linearRegression
model = LinearRegression()
model.fit(train, y_train)
print("훈련 세트 점수: {:.2f}".format(model.score(train, y_train)))
print("테스트 세트 점수: {:.2f}".format(model.score(test, y_test)))
print("사용한 특성의 수: {}".format(   np.sum( model.coef_ != 0 )   ))

#polynomial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(youtube_view)
youtube_view.shape, x_poly.shape
train,test,y_train,y_test=train_test_split(x_poly,views, test_size=0.2,shuffle=False)

model = LinearRegression()
model.fit(train, y_train)
print("훈련 세트 점수: {:.2f}".format(model.score(train, y_train)))
print("테스트 세트 점수: {:.2f}".format(model.score(test, y_test)))
print("사용한 특성의 수: {}".format(   np.sum( model.coef_ != 0 )   ))

#randomforest
RF = RandomForestRegressor()
RF.fit(train,y_train)
print("훈련 세트 점수: {:.2f}".format(RF.score(train, y_train)))
print("테스트 세트 점수: {:.2f}".format(RF.score(test, y_test)))

#gridsearchCV
nEstimator = [140,160,180,200,220]
depth = [10,15,20,25,30]
hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
grid = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
grid.fit(train,y_train)
print("Best HyperParameter: ",grid.best_params_)
print(grid.best_score_)
scores = grid.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.colorbar()
plt.xticks(np.arange(len(nEstimator)), nEstimator)
plt.yticks(np.arange(len(depth)), depth)
plt.title('Grid Search r^2 Score')
plt.show()
maxDepth=grid.best_params_['max_depth']
nEstimators=grid.best_params_['n_estimators']
