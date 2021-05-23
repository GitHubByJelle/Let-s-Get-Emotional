# CV_assignment2
Assignment 2 of CV

## File to train network with terminal
### train.py
Run "train.py -h" for help with input.
-network (architecture to use) is required.
Example: "   python train.py -network JNet1 -loader small    " (Don't forget to include -network and -loader).

Please note that; our best performing model is "python train.py -network JNet18 -loader balancedtransform"
However, it requires longer runtimes compared to JNet1 with small loader.

### camera.py
Run "camera.py" to run file. That's it!

## Files used during project (using IDE to run files)
### main.py
Is used to train the networks.
Note, the data isn't added (because it's to large to push on GitHub)
To change parameters (e.g. network architecture, learning rate, loadertype, nepochs, etc), go into main.py and change them at the top of __main__

### visualiser.py
Is used to make functions for visualisations

### loaders.py
Is used to load in the data (and oversample / augment)

### nets.py
Is used to make architectures. Almost all the architectures made during this assignment can be found there.

### NetworkAnalysis.py
Is used to analyse the network (e.g. performance, convolution layers, etc.).
Uncomment parts u want to use (as default everything is commented, so you could only select what you want).

### VisualisationsReport.py
Is used to make visualisations for the report

### Finetuning.py
Is used to finetune an existing network (e.g. VGG, ResNet, SqueezeNet, etc).

###  camera.py
Classifies the emotions of the user by using the camera!
Just run the file to test it!

## Folders
### data
Place the FER2013.csv file in the FER folder.

### camera
Holds file needed for facial detection.

### networks
Hold all networks trained and visualisations of training


