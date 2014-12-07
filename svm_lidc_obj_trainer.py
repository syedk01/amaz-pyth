#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the svm model for objective features


import csv
import numpy as np
import pickle
import os
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib

X = np.empty([0, 9]) #Feature array
Y = [] #Target class
f = open('obj_master_file.csv', 'rb') #Reading the training data
reader = csv.reader(f)

for row in reader:
	#Popping last 3 values, as first 9 objective features are considered for now
	Y.append(row.pop())
	row.pop()
	row.pop()
	row.pop()
	X=np.vstack((X, row))

clf = svm.SVC()
clf.fit(X, Y)
joblib.dump(clf, os.environ["PAC_HOME"]+'/paccloud/data/LIDC/objective/svm_lidc_obj_train_data.pkl') #Saving the objective training file

#scores=cross_validation.cross_val_score(clf, X, np.array(Y), cv=10)
#print "Scores"
#print scores
