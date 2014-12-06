#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file trains the svm model for subjective features

#!/usr/bin/python

import csv
import numpy as np
import os
#import sklearn as skl
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
X=np.empty([0, 8]) #Feature array
Y=[] #Target class
f=open('sub_master_file.csv', 'rb')#Reading the training data
reader=csv.reader(f)

for row in reader:
	Y.append(row.pop())
	X=np.vstack((X, row))


clf=svm.SVC()
clf.fit(X, Y)

joblib.dump(clf, os.environ["PAC_HOME"]+'/paccloud/data/LIDC/subjective/svm_lidc_sub_train_data.pkl') #Saving the objective training file


#scores=cross_validation.cross_val_score(clf, X, np.array(Y), cv=10)

#print scores
