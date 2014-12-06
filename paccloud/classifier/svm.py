#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#This file load svm training data and predicts

#!/usr/bin/python

from sklearn import svm
from sklearn.externals import joblib

def load(file):
	clf=joblib.load(file)
	return clf

def predict(clf, X):
	p=clf.predict(X)
	return p

	
