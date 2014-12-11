#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the knn model for subjective features


import csv
import numpy as np
import os
from pybrain import datasets
from pybrain.datasets import classification
from pybrain import tools
from pybrain.tools import shortcuts
from pybrain.tools import xml
from pybrain.tools.xml import networkwriter
from pybrain import supervised
from pybrain.supervised import trainers
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn import decomposition

hidden_layer = 10

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

kf = KFold(33315, n_folds = 10)

scores = []
stdv = []

for train, test in kf:
	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test] # Splitting the training data
	
	ds_t = classification.ClassificationDataSet(X_train.shape[1], 1, 5) 
	net_t = shortcuts.buildNetwork(X_train.shape[1], hidden_layer, 1, bias = True)

	X_train =  preprocessing.scale(X_train.astype(np.float64)) # Feature scaling and normalization
	X_train  = preprocessing.normalize(X_train.astype(np.float64), norm = 'l1')
	pca = decomposition.PCA( whiten = True ); #PCA decomposition
	pca.fit(X_train)
	X_train = pca.transform(X_train)

	X_test = preprocessing.scale(X_test.astype(np.float64))
	X_test  = preprocessing.normalize(X_test.astype(np.float64), norm = 'l1')
	X_test = pca.transform(X_test)
	
	for i in xrange(X_train.shape[0]):
		ds_t.addSample([X_train[i][n] for n in xrange(X_train.shape[1])], [Y_train[i]])

	trainer_t = trainers.BackpropTrainer( net_t, ds_t )
	trainer_t.train()
	
	P = []
	for i in xrange(X_test.shape[0]):
		P.append(0)
		P[i] = net_t.activate([X_test[i][n] for n in xrange(X_test.shape[1])])[0]

	P = np.array(P)

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

X =  preprocessing.scale(X.astype(np.float64)) # Feature scaling and normalization
X = preprocessing.normalize(X.astype(np.float64), norm = 'l1')
pcag = decomposition.PCA( whiten = True ); #PCA decomposition
pcag.fit(X)
X = pcag.transform(X)


ds = classification.ClassificationDataSet(X.shape[1], 1, 5)

net = shortcuts.buildNetwork(X.shape[1], hidden_layer, 1, bias = True)

for i in xrange(X.shape[0]):
	ds.addSample([X[i][n] for n in xrange(X.shape[1])], [Y[i]])

trainer = trainers.BackpropTrainer( net, ds )
trainer.train()



networkwriter.NetworkWriter.writeToFile(net, os.environ["PAC_HOME"]+'/paccloud/data/LIDC/objective/nnet_lidc_obj_train_data.xml') #Saving the objective training file

