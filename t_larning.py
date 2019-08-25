import csv
import numpy as np
import scipy

from sklearn import svm



#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor

#est = LinearRegression(fit_intercept=False)

clf = svm.SVC(gamma=0.001, C=100.)

#est = RandomForestRegressor()


x_file = "x_train.csv"
y_file = "y_train.csv"

xt_file = "x_test.csv"

yt_file = "y_test.csv"

remove_cols = [7, 8 , 14,  27, 31, 47, 52,70,  71, 77, 88,   95, 105,108,  118, 119,126,  136, 139,  142, 145,  168, 169, 184, 194, 200,201,  209, 217]
#with open(x_file,'r') as dest_f:  
#    data_iter = csv.reader(dest_f, 
#                           delimiter = ";", 
#                           quotechar = '"')
#    data_x = [data for data in data_iter]

#with open(y_file,'r') as y_f:
#    data_y_iter = csv.reader(y_f, 
#                           delimiter = ";", 
#                           quotechar = '"')
#    data_y = [data for data in data_y_iter]


data_x = np.genfromtxt(x_file, dtype=float, delimiter=';') 
data_y = np.genfromtxt(y_file, dtype=float, delimiter=';') 
data_xt = np.genfromtxt(xt_file, dtype=float, delimiter=';') 

##offset1 =1
#
#for i in remove_cols:
#    data_x = scipy.delete(data_x, i-offset1, 1)  # delete 6th column
#    data_xt = scipy.delete(data_xt, i-offset1, 1)  # delete
#    offset1 += 1


#print (data_x.tolist())


#with open(xt_file,'r') as xt_f:
#    data_xt_iter = csv.reader(xt_f,  dtype=float,
#                           delimiter = ";", 
##                           quotechar = '"')
#    data_xt = [data for data in data_xt_iter]
#print (data_xt)

#with open(yt_file,'w') as ot_f:
#    data_yt_iter = csv.writer(xt_f, 
#                           delimiter = ";", 
#                           quotechar = '"')
#    data_yt = [data for data in data_xt_iter]



#y2  = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
#print (data_xt)
##corr_x=data_x.corr()

clf.fit(data_x, data_y)
##z=est.coef_   # access coefficients


#est.score(data_xt, y2)
xxx = clf.predict(data_xt)
y2 = xxx.tolist()
#print (y2)
#print (type(xxx))
for i in y2:
   print(int (i))




#print data_x[0][33]
#print data_y
