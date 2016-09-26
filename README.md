# musicComposition
This is the project of LSTM-BASED MUSIC COMPOSITION SYSTEM WITH DYNAMIC MUSIC REPRESENTATION.

## 1. Usage
### 1.1 Environment
The code has been tested under Anaconda2, and we recommend using Anaconda2 as your python environment.

For other essencial library, you may refer to requirements.txt, and using this command to install 
(this may includes library more than needed).

    $ pip install -r requirements.txt
### 1.2 Parse dataset
Use this command to parse the sheet music stored in */dataset/demo.abc* to program readable music form */dataset/demo.txt*.

    $ python generate_data.py 
    
This process should be finished in less than 1 minute.
### 1.3 Train and compose music
If you have cuda installed, use this command to train and compose music with GPU acceleration.

    $ THEANO_FLAGS=device=gpu,floatX=float32 python train.py
Or just omit the flags and train with CPU, but the process will be very slow.

    $ python train.py
The generated music will be stored in */composed_melody/demo*.

The whole training and composing process is less than 1 hour on GTX1060 computer for demo code.

## 2. Train your own model
1. Collect sheet music in [abc notation](http://abcnotation.com/), and save it in */dataset/somedataset.abc*.

2. Replace all the existing "demo" with "somedataset" in *read_abc.py generate_data.py and train.py*.

3. Do 1.2 and 1.3 to train your own model and compose music.

## 3. Composed music

We've trained several dataset and created music of different styles. And these can all be accessed in */composed_melody/*

1. demo: the model is trained from demo dataset （805 songs of various types), the composed music is triggerred with small pieces of melody not in the dataset.
2. france: the model is trained from french traditional music dataset （1000 french traditional songs), the composed music is triggerred with small pieces of melody not in the dataset.
3. mixed: the model is trained from mixed music dataset （20000 songs of various types), the composed music is triggerred with small pieces of melody not in the dataset.
