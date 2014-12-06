amaz-pyth v 40.1
=================
Installation Guide:
===================

The following instructions are for Debian based systems(Ubuntu, Debian, Mint etc.) only and tested for Ubuntu 14.04.

*(Recommended) Update your repo with "sudo apt-get update"
*Make the setup.sh executable with "sudo chmod +x setup.sh"
*Run "sudo ./setup.sh". It would install gcc, python, node js and other dependencies if you don't have them already installed. It might take some time.
*After the setup finished do ". ~/.bashrc". This will update your environment variables.
* Run ./svm_lidc_sub_trainer.py and ./svm_lidc_ob_trainer.py one after another. The 2nd one will take a lot of time. Be patient.
* Run "sudo nodjes server.js 0". Now you can access the dicom viewer with http://<your-ip/web address-here>:8080 and start annotating.

TODO:
=====

1. Make the dicom loaded from url usable with pyton. Load from google drive url.
2. Currently it works on 9 objective features, make use of all 12.
3. Implement nnet and knn.
4. Adding model chooser(svm/knn/nnet) on the front end gui.

Known Issues:
=============

1. Can't load multislice dicom images.
2. Can't laod from google drive url.
3. Struggles to handle large dicom files.
