#!/usr/bin/python

# Training all classifier

import os

classifier_type = ['svm', 'knn', 'nnet']
target_type = ['lidc']
feature_type = ['sub', 'obj']

for classifier in classifier_type:
	for target in target_type:
		for features in feature_type:
			print "Training " + classifier + " for " + target + " with " + features + " features...... "
			os.system("./"+classifier + "_" + target + "_" + features + "_trainer.py")		
