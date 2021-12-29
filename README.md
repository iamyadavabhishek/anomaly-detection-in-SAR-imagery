# anomaly-detection-in-SAR-imagery
Anomaly detected using aerial imagery can be of many types like unknown ship in docks, or a unidentified aircraft at the airport and other similar anomalies. In this project we've mainly focused on ship dataset to detect all the ships in the waterbody using keras-retinanet.

# pre-requisites
firstly you should have installed nvidia cuda toolkit so your jupyter notebook can access your gpu resources.
to have your jupter notebook run on gpu, you can follow this link: https://www.techentice.com/how-to-make-jupyter-notebook-to-run-on-gpu/
you can also have a linux machine using gpu. any will work.

# Libraries
* os
* numpy
* tensorflow
* sklearn
* cv2
* PIL
* h5py

![image](https://user-images.githubusercontent.com/94900416/147631648-fab908de-56f5-4b88-9850-dc66893854d6.png)

# fizyr retinanet
Fizyr has fortunately implemented the keras retinanet model and made it available to everyone who wants to use it so we don't have to implement the model from the scratch. we can fine tune the model so it works accordingly!

![image](https://user-images.githubusercontent.com/94900416/147631868-38577ea9-9093-4407-aab6-8917253678af.png)

