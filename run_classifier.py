#!/usr/bin/python

import sys
import logging as log
import os
from paccloud.classifier import svm
import numpy as np
from paccloud.dicomutils import reader
from paccloud.dicomutils import utils


classifier_type=sys.argv[1].lower()
target_type=sys.argv[2].upper()
sub_values=sys.argv[3]
result_file=sys.argv[4]
dicom_file = sys.argv[5]
p = sys.argv[6]

sub_values=sub_values.replace('[', '')
sub_values=sub_values.replace(']', '')
sub_values=sub_values.replace(' ', '')

#log.info("I am running!")


pklfilesub=os.path.dirname(os.path.abspath(__file__))+'/paccloud/data/'+target_type+'/subjective/'+classifier_type+'_'+target_type.lower()+'_sub_train_data.pkl'	
pklfileobj=os.path.dirname(os.path.abspath(__file__))+'/paccloud/data/'+target_type+'/objective/'+classifier_type+'_'+target_type.lower()+'_obj_train_data.pkl'

Xsub=sub_values.split(',')
image = reader.read(dicom_file)
Xobj = utils.calculate_obj_features(image, p)

psub = None
pobj = None
clfsub = None
clfobj = None
pobj = None

if classifier_type=='svm':
	clfsub=svm.load(pklfilesub)
	clfobj=svm.load(pklfileobj)
	psub=svm.predict(clfsub, np.array(Xsub))
	pobj = svm.predict(clfobj, np.array(Xobj))
pobj = float(pobj[0])/5
psub = float(psub[0])/5
f=open(result_file, 'w')
f.write(str(pobj))
f.write(',')
f.write(str(psub))
f.close()
