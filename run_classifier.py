#!/usr/bin/python

#Author: Rifat Mahmud(rftmhmd@gmail.com)
#Developed for: Syed Khalid, Pacific Cloud(syedk@pacificloud.com)
#run_classifier.py: This program will load the trained file and based on it, predict a value of malignancy of 1-5 in both sub and obj cases
#The program will be called by server when the classify button is clicked by the user on the GUI.

import sys
import logging as log
import os
from paccloud.classifier import svm
from paccloud.classifier import knn
import numpy as np
from paccloud.dicomutils import reader
from paccloud.dicomutils import utils


classifier_type = sys.argv[1].lower() #Fist argument will be one of these svm/knn/nnet
target_type = sys.argv[2].upper() # LIDC/INBREAST
sub_values = sys.argv[3] #Subjective rating vaules by users
result_file = sys.argv[4] #Where the predicted value would be stored
dicom_file = sys.argv[5] #Dicom file to be processed
p = sys.argv[6] #Pixel boundary of the annotated area

#Extracting the sub values from the string
sub_values = sub_values.replace('[', '')
sub_values = sub_values.replace(']', '')
sub_values = sub_values.replace(' ', '')

#Loading the pkl files(training files)
pklfilesub = os.path.dirname(os.path.abspath(__file__))+'/paccloud/data/'+target_type+'/subjective/'+classifier_type+'_'+target_type.lower()+'_sub_train_data.pkl'	
pklfileobj = os.path.dirname(os.path.abspath(__file__))+'/paccloud/data/'+target_type+'/objective/'+classifier_type+'_'+target_type.lower()+'_obj_train_data.pkl'

Xsub = sub_values.split(',')

image = reader.read(dicom_file)#Reading dicom files

Xobj = utils.calculate_obj_features(image, p) #Calculating objective features

psub = None
pobj = None
clfsub = None
clfobj = None
pobj = None

if classifier_type=='svm':
	clfsub = svm.load(pklfilesub)
	clfobj = svm.load(pklfileobj)
	psub = svm.predict(clfsub, np.array(Xsub))
	pobj = svm.predict(clfobj, np.array(Xobj))
elif classifier_type=='knn':
        clfsub = svm.load(pklfilesub)
        clfobj = svm.load(pklfileobj)
        psub = svm.predict(clfsub, np.array(Xsub))
        pobj = svm.predict(clfobj, np.array(Xobj))
		
pobj = float(pobj[0])/5#Objective prediction value
psub = float(psub[0])/5#Subjective prediction value
#Writing the prediction
f=open(result_file, 'w')
f.write(str(pobj))
f.write(',')
f.write(str(psub))
f.close()
