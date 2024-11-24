# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Command line applications train.py and predict.py
For command line applications there is an option to select either Alexnet or VGG16 models.

Following arguments mandatory or optional for train.py

'data_dir'. 'Provide data directory. Mandatory argument', type = str
'--save_dir'. 'Provide saving directory. Optional argument', type = str
'--arch'. 'Vgg16 can be used if this argument specified, otherwise Alexnet will be used', type = str
'--lrn'. 'Learning rate, default value 0.001', type = float
'--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int
'--epochs'. 'Number of epochs', type = int
'--GPU'. "Option to use GPU", type = str
Following arguments mandatory or optional for predict.py

'image_dir'. 'Provide path to image. Mandatory argument', type = str
'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str
'--top_k'. 'Top K most likely classes. Optional', type = int
'--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
'--GPU'. "Option to use GPU. Optional", type = str