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


len_x = data_x.shape [1]
#print (len_x)

for i in range(len_x):
    len_x = data_x.shape [1]
    if (i >= len_x):
        break
#    print (len_x)
    correl_x_y=float(np.corrcoef(data_y, data_x[:,i])[0,1])
    if (correl_x_y < 0.99999)  and  (correl_x_y  > -0.999999):
#        print (correl_x_y)
        data_x = scipy.delete(data_x, i, 1)  # delete 6th column
        data_xt = scipy.delete(data_xt, i, 1)  # delete



len_x = data_x.shape [1]
print ("len_x = %d" % len_x)

for i in range(len_x):
 len_x = data_x.shape [1]
 if (i >= len_x):
        break
 for j in range(i+1,len_x):
    len_x = data_x.shape [1]
    if (j >= len_x):
        break
#    print (len_x)
#    print (" i = %d, j = %d" %(i,j))
#    print (np.corrcoef(data_x[:,i], data_x[:,j]))
    correl_x_x=float(np.corrcoef(data_x[:, i], data_x[:,j])[0,1])
    if (correl_x_x >  0.95) or  (correl_x_x  < -0.95):
        print (correl_x_x)
        data_x = scipy.delete(data_x, j, 1)  # delete 6th column
        data_xt = scipy.delete(data_xt, j, 1)  # delete








#est = RandomForestRegressor(n_estimators=200, max_depth=None,
for spliters  in range(100 , 800, 25):

 est = RandomForestClassifier( n_estimators=spliters, criterion='gini', max_depth=None, min_samples_split=4 ,min_samples_leaf=1,  min_weight_fraction_leaf=0.0, 
			      max_features='sqrt', max_leaf_nodes=None, min_impurity_split=1e-5,  bootstrap=True, oob_score=True, n_jobs=-1, 
			      random_state=28, verbose=0, warm_start=False, class_weight=None)

 est.fit(data_x, data_y)

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

#for i in y2:
#  print(int (round(i,0)))

