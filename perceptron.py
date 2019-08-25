import csv
import numpy as np
import scipy


from scipy.cluster import hierarchy

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import preprocessing



import sys
sys.setrecursionlimit(10000)


x_file = "x_train.csv"
y_file = "y_train.csv"

xt_file = "x_test.csv"

yt_file = "y_test.csv"


data_x = np.genfromtxt(x_file, dtype=float, delimiter=';') 
data_y = np.genfromtxt(y_file, dtype=float, delimiter=';') 
data_xt = np.genfromtxt(xt_file, dtype=float, delimiter=';') 

data_x = preprocessing.normalize(data_x)
data_X = preprocessing.scale(data_x)



data_xt = preprocessing.normalize(data_xt)
data_xt = preprocessing.scale(data_xt)


rf = RandomForestClassifier(n_estimators=625, criterion='gini', max_depth=None, min_samples_split=4 ,min_samples_leaf=1,  min_weight_fraction_leaf=0.0,
                              max_features='sqrt', max_leaf_nodes=None, min_impurity_split=1e-5,  bootstrap=True, oob_score=True, n_jobs=-1,
                              random_state=28, verbose=0, warm_start=False, class_weight=None)




#est = CalibratedClassifierCV (rf, method='isotonic', cv=5)

est=Perceptron(fit_intercept=False, n_iter=100, shuffle=False)

est.fit(data_x, data_y)

scores = cross_val_score(est, data_x, data_y )
print ("score mean  =    %f"  % scores.mean() )


#Z = hierarchy.linkage(data_x, 'single')


#plt.figure()


#dn = hierarchy.dendrogram(Z)

#plt.show()


#print (est.score (data_x,data_y))

xxx=est.predict(data_xt)

y2 = xxx.tolist()

#scores = cross_val_score(est, data_x, data_y, cv=5, scoring='accuracy')
#print("Accuracy: %0.2f (+/- %0.2f) " % (scores.mean(), scores.std())) 

#print ("AUC-ROC (oob) = %0.2f" % est.oob_score_)

for i in y2:
  print(int (round(i,0)))

