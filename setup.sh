#!/bin/bash

apt-get --assume-yes install gcc
apt-get --assume-yes install g++
apt-get --assume-yes install gfortran
apt-get --assume-yes install python
apt-get --assume-yes install python-dev
apt-get --assume-yes install python-scipy
apt-get --assume-yes install python-pip

pip install --upgrade numpy
pip install scikit-learn
pip install image
pip install scikit-image
pip install pydicom

apt-get --assume-yes install nodejs
apt-get --assume-yes install npm

npm install connect
npm install formidable
npm install fs
npm install generate-key
npm install http
npm install multipart
npm install querystring
npm install request
npm install serve-static
npm install sys
npm install url
npm install util


bashrc=~/.bashrc

if [! -f "$bashrc"]; then
touch "$bashrc"
fi

mkdir OUTPUT
mkdir FILE
_cwd="$(pwd)"

echo "export PAC_HOME="$_cwd"">txt
cat txt>>"$bashrc"

rm -f txt

chmod -R 777 site
mkdir paccloud/data
mkdir paccloud/data/LIDC
mkdir paccloud/data/INBREAST
mkdir paccloud/data/LIDC/subjective
mkdir paccloud/data/LIDC/objective
mkdir paccloud/data/INBREAST/subjective
mkdir paccloud/data/INBREAST/objective
chmod +x *.py

bash

./svm_lidc_sub_trainer.py
./svm_lidc_obj_trainer.py
