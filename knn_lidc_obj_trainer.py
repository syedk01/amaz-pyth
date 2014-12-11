#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the knn model for subjective features


import csv
import numpy as np
import os
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn import decomposition

X = np.empty([0, 12]).astype(np.float64) #Feature array
Y = [] #Target class
f = open('obj_master_file.csv', 'rb')#Reading the training data
reader = csv.reader(f)

for row in reader: # Reading the csv row by row
	Y.append(row.pop())
#	row.pop()
	X=np.vstack((X, row))

Y = np.array(Y).astype(np.float64)

Y = (Y - 1)/4 # Normalizing the target array. I case of LIDC subjective, the value is 1-4

clf = neighbors.KNeighborsClassifier(n_neighbors = 50, weights = 'distance')

kf = KFold(33315, n_folds = 10)

scores = []
stdv = []

for train, test in kf:
	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test] # Splitting the training data
	clft = neighbors.KNeighborsClassifier( n_neighbors = 2 ) 

	X_train =  preprocessing.scale(X_train.astype(np.float64)) # Feature scaling and normalization
	X_train  = preprocessing.normalize(X_train.astype(np.float64), norm = 'l1')
	pca = decomposition.PCA( whiten = True ); #PCA decomposition
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	clft.fit(X_train, Y_train)
	X_test = preprocessing.scale(X_test.astype(np.float64))
	X_test  = preprocessing.normalize(X_test.astype(np.float64), norm = 'l1')
#	X_test = pca.transform(X_test)

	P = clft.predict(X_test)

	for e in xrange(P.shape[0]):
		if P[e]<0:
			P[e] = 0
		elif P[e]>0:
			P[e] = 1

	#print type(P), P.shape, type(Y_test), Y_test.shape	
	diff = P.astype(np.float64) - Y_test.astype(np.float64)
	diff = np.fabs(diff) # Calculating the absoulte difference between predicted value and real value
	print np.sum(diff)
	#print type(diff), diff	
	scores.append(np.sum(diff)/Y_test.shape[0])
	#stdv.append(np.std(diff))

print scores 
print np.std(np.array(scores))
#scores=cross_validation.cross_val_score(clf, X, np.array(Y), cv=10, scoring = 'f1')

#print scores

clf.fit(X, Y)

joblib.dump(clf, os.environ["PAC_HOME"]+'/paccloud/data/LIDC/objective/knn_lidc_obj_train_data.pkl') #Saving the objective training file

