# RetiNet - Automated AMD identification in OCT volumetric data

Companion software for the relevant paper. This python script will download, compile and cross-validate RetiNet against the dataset mentioned in the paper. The original version of the dataset can be found here: http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm

## Usage:
This program requires a working installation of Python 3.2 or newer and Keras 1.0.8 or newer. Nvidia Cuda and CuDNN must also be installed for GPU acceleration.

A requirements.txt file is provided for convenience. Given a working installation of Python3, this can be installed as follows:
```
pip3 install --upgrade -r requirements.txt
``` 

Once the dependencies are satisfied, you can execute this simply by typing:
```
python3 retinet.py
```

The required dataset will be downloaded, extracted and evaluated automatically. 
