#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the svm model for subjective features


import csv
import numpy as np
import os
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn import decomposition

X = np.empty([0, 12]).astype(np.float64) #Feature array
Y = [] #Target class
f = open('obj_master_file.csv', 'rb')#Reading the training data
reader = csv.reader(f)

l = 0
for row in reader: # Reading the csv row by row
	Y.append(row.pop())
#	row.pop()
	X=np.vstack((X, row))
	l = l+1
	print l
	#if l == 10:
		#break

Y = np.array(Y).astype(np.float64)
X = np.array(X).astype(np.float64)

Y = (Y - 1)/4 # Normalizing the target array. I case of LIDC subjective, the value is 1-4

clf = svm.SVC(gamma = 10)

kf = KFold(33315, n_folds = 10)

scores = []
stdv = []

l = 0
for train, test in kf:
	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test] # Splitting the training data
	clft = svm.SVC(gamma = 10)
	mean = np.mean(X_train, axis = 0, dtype = np.float64) #Getting mean of each column
	scale = np.amax(np.absolute(X_train), axis = 0 ) #Getting the largest value of each column
	l = l + 1 
	print l
     
 
	for i in xrange(X_train.shape[1]):
		X_train[:, i] = ((X_train[:, i] - mean[i])/scale[i]).astype(np.float64) #Substract from each value the mean of the column and divide it by the largest value.
																				# Thus values remain inside -1 to 1
	pca = decomposition.PCA(whiten = True); #PCA decomposition
	pca.fit(X_train.astype(np.float64))
	X_train = pca.transform(X_train.astype(np.float64)) #Apply PCA transformation
#	print X_train
	clft.fit(X_train, Y_train)
 
	for i in xrange(X_test.shape[1]):
		X_test[:, i] = ((X_test[:, i] - mean[i])/scale[i]).astype(np.float64)
#	X_test = preprocessing.scale(X_test.astype(np.float64))
  
	X_test = pca.transform(X_test.astype(np.float64))
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
#print np.std(np.array(scores))
#scores=cross_validation.cross_val_score(clf, X, np.array(Y), cv=10, scoring = 'f1')

#print scores

mean = np.mean(X_train, axis = 0, dtype = np.float64) #Getting mean of each column
scale = np.absolute(np.amax(X_train, axis = 0 )) #Getting the largest value of each column
 
#print type(mean), mean.shape
 
for i in xrange(X.shape[1]):
	X[:, i] = ((X[:, i] - mean[i])/scale[i]).astype(np.float64) #Substract from each value the mean of the column and divide it by the largest value.
						

pcag = decomposition.PCA( whiten = True ); #PCA decomposition
pcag.fit(X)
X = pcag.transform(X)


clf.fit(X, Y)

joblib.dump(clf, #os.environ["PAC_HOME"]+'/
            'paccloud/data/LIDC/objective/svm_lidc_obj_train_data.pkl') #Saving the objective training file

