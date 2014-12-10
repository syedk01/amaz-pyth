#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the knn model for subjective features


import csv
import numpy as np
import os
#import sklearn as skl
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
X = np.empty([0, 8]) #Feature array
Y = [] #Target class

f = open('sub_master_file.csv', 'rb')#Reading the training data
reader = csv.reader(f)

for row in reader:
	Y.append(row.pop())
	X = np.vstack((X, row))


clf = neighbors.KNeighborsClassifier(n_neighbors = 25)

kf = KFold(6470, n_folds = 10)

scores = []
stdv = []

for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        clft = neighbors.KNeighborsClassifier(n_neighbors = 25)
        clft.fit(X_train, Y_train)

        P = clft.predict(X_test)
        #print type(P), P.shape, type(Y_test), Y_test.shape
        diff = P.astype(np.float64) - Y_test.astype(np.float64)
#       diff = np.absolute(diff)
        #print type(diff), diff
        scores.append(np.sum(diff)/Y_test.shape[0])
        #stdv.append(np.std(diff))

print scores
print np.std(np.array(scores))


#scores=cross_validation.cross_val_score(clf, X, np.array(Y), cv=10)

#print scores

clf.fit(X, Y)

joblib.dump(clf, os.environ["PAC_HOME"]+'/paccloud/data/LIDC/subjective/knn_lidc_sub_train_data.pkl') #Saving the objective training file

