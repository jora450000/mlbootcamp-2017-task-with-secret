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


import sys
sys.setrecursionlimit(10000)


#est = LinearRegression(fit_intercept=False)

x_file = "x_train.csv"
y_file = "y_train.csv"

xt_file = "x_test.csv"

yt_file = "y_test.csv"


data_x = np.genfromtxt(x_file, dtype=float, delimiter=';') 
data_y = np.genfromtxt(y_file, dtype=float, delimiter=';') 
data_xt = np.genfromtxt(xt_file, dtype=float, delimiter=';') 


remove_cols = [7, 8 , 14,  27, 31, 47, 52,70,  71, 77, 88,   95, 105,108,  118, 119,126,  136, 139,  142, 145,  168, 169, 184, 194, 200,201,  209, 217]







offset1 =1

for i in remove_cols:
    data_x = scipy.delete(data_x, i-offset1, 1)  # delete 6th column
    data_xt = scipy.delete(data_xt, i-offset1, 1)  # delete
    offset1 += 1




#est = RandomForestRegressor(n_estimators=200, max_depth=None,
for spliters  in range(0 , 1, 1):

 est = RandomForestClassifier( n_estimators=625, criterion='gini', max_depth=None, min_samples_split=4 ,min_samples_leaf=1,  min_weight_fraction_leaf=0.0, 
			      max_features='sqrt', max_leaf_nodes=None, min_impurity_split=1e-5,  bootstrap=True, oob_score=True, n_jobs=-1, 
			      random_state=28, verbose=0, warm_start=False, class_weight=None)

 est.fit(data_x, data_y)

 if (est.oob_score_ > 0.57):
    print  ("AUC-ROC for %d (oob) = %0.6f" % (spliters, est.oob_score_))






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

