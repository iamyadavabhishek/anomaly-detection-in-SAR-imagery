# anomaly-detection-in-SAR-imagery
Anomaly detected using aerial imagery can be of many types like unknown ship in docks, or a unidentified aircraft at the airport and other similar anomalies. In this project we've mainly focused on ship dataset to detect all the ships in the waterbody using keras-retinanet.

# pre-requisites
firstly you should have installed nvidia cuda toolkit so your jupyter notebook can access your gpu resources.<br>
to have your jupter notebook run on gpu, you can follow this link: https://www.techentice.com/how-to-make-jupyter-notebook-to-run-on-gpu/<br>
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

# preparing dataset
firstly we need to seperate our dataset into train,test and validation.<br>
1) strip the images of their extensions and store the names in an array<br>
```
os.chdir(r"D:\notebooks\RetinanetTutorial/SAR Dataset/images")
data=np.empty(len(os.listdir()),dtype='object')
for i,j in zip(os.listdir(),range(len(os.listdir()))):
    data[j]="{}".format(i.strip('.png'))
print(data)
```
2) use sklearn to split this data in train,val and test:
```
train,test = train_test_split(data,test_size=0.1,shuffle=True,random_state=8)
train,val = train_test_split(train,test_size=0.2,shuffle=True,random_state=8)
print('train:{}, val:{}, test:{}'.format(len(train),len(val),len(test)))
```
3) store these in train.txt,val.txt and test.txt respectively:
```
os.chdir(r"D:\notebooks\RetinanetTutorial/SAR Dataset/ImageSets/Main")
trainf = open("train.txt","w+")
valf = open("val.txt","w+")
testf = open("test.txt","w+")
for i in train:
    trainf.write("{}\n".format(i))
for i in val:
    valf.write("{}\n".format(i))
for i in test:
    testf.write("{}\n".format(i))
trainf.close()
valf.close()
testf.close()
```
4) the original dataset was in the form of png. for our model we have to convert it into jpeg format. we use PIL library to convert all the images into jpeg:
```
os.chdir(r"D:\notebooks\RetinanetTutorial\SAR_Dataset\images")
for i in os.listdir():
    img_png = Image.open("D:/notebooks/RetinanetTutorial/SAR_Dataset/images/{}.png".format(i.strip('.png')))
    img_png = img_png.convert('RGB')
    img_png.save("D:/notebooks/RetinanetTutorial/SAR_Dataset/JPEGImages/{}.jpg".format(i.strip('.png')))
 ```
5) `!mkdir D:\notebooks\RetinanetTutorial\Output\Snapshots` to make a directory to store your model snapshots

# 
