import csv
import numpy as np
import scipy

from sklearn.ensemble import GradientBoostingClassifier



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

#random_state=0, hidden_layer_sizes=[10]



gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(data_x, data_y)


#est.score(data_xt, y2)
xxx = gbrt.predict(data_xt)
y2 = xxx.tolist()
##print (y2)
#print (type(xxx))
for i in y2:
   print(int (i))




#print data_x[0][33]
#print data_y
