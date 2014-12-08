#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the knn model for subjective features


import csv
import numpy as np
import os
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.externals import joblib

X = np.empty([0, 12]) #Feature array
Y = [] #Target class
f = open('obj_master_file.csv', 'rb')#Reading the training data
reader = csv.reader(f)

for row in reader:
	Y.append(row.pop())
	X=np.vstack((X, row))


clf = neighbors.KNeighborsClassifier()
clf.fit(X, Y)

joblib.dump(clf, os.environ["PAC_HOME"]+'/paccloud/data/LIDC/objective/knn_lidc_obj_train_data.pkl') #Saving the objective training file


#scores=cross_validation.cross_val_score(clf, X, np.array(Y), cv=10)

#print scores
