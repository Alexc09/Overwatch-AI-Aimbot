# Overwatch-AI-Aimbot

Object Detection with YOLO/MOBILENET architecture

~ Coming Soon

## preprocessImg.py
Used to rename image filenames in ascending order 
(E.g 1.png, 2.png, etc.)
```
python preprocessImg.py [--filepath]
```
--filepath: Folder containing images

E.g: `python processImg.py Workspace/images/screenshots`


## train.py
Used to process data train model
- Generated Tfrecords & labelmap will be in Workspace/annotataions
- Model weights will be saved to Workspace/my_model/{{CUSTOM_MODEL_NAME}}/


Will automatically
- Split images into train/test folders
- Edit pipeline.config file
- Create tfrecords from images
```
python train.py [-L --labels] [-E --epochs] [-N --name] [-T --split] [-s --trainSize]
```

>--labels: (str) Labels in json format. Must follow {'name': ..., 'id': ...} convention, where 'name' is the label given in labelImg.py tool.  
> Separate labels using '***' triple asterix (E.g "{...}***{...}" )

>--epochs: (int) Number of training epochs to run

>--name: (str) Name of custom model 

>--split: (str) `"yes"` or `"no"` (Defaults to 'no'). If "yes", will automatically split images & annotations in /screenshots folder, to /train and /test folder using trainSize 

>--trainSize: (float) Training size (0-1, defaults to 1)

E.g: `python train.py -L "{'name': 'head', 'id': 1 }***{'name': 'body', 'id': 2 }" -E 50 -N "overwatch_mobnet" -T yes -S 0.7`
Internally, this will create:
- label=[{'name': 'head', 'id': 1 }, {'name': 'body', 'id': 2 }]
- Run for 50 epochs
- Create a model named "Overwatch_mobnet"
- Split the files in /screenshot folder, with 70% going to /train and the remaining 30% going to /test
- Run training for the model, where outputs can be found in my_models/overwatch_mobnet/
