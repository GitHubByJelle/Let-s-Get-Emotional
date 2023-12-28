# Let's Get Emotional
This project is part of an group assignment for the course Computer Vision for my Master's in Artificial Intelligence. The project involves developing a Convolutional Neural Network (CNN) to achieve optimal performance on the [Facial Expression Recognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition) (FER2013) dataset obtained from the [Kaggle challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) on representation learning for facial expression recognition. 

In order to create the "best-performing" model, different architectures have been implemented by iteratively modifying the architecture and the parameters (learning rate, number of epochs, etc.) used during training to achieve the highest possible accuracy. Since the FER2013 dataset is unbalanced, data augmentation and oversampling have been used to solve this issue. To compare the performance of our architectures with existing architectures, several models were also finetuned: [ResNet](https://arxiv.org/abs/1512.03385), [AlexNet](https://arxiv.org/abs/1404.5997), [VGG](https://arxiv.org/abs/1409.1556), [SqueezeNet](https://arxiv.org/abs/1602.07360), [DenseNet](https://arxiv.org/abs/1608.06993).

In addition to the assigned task, an extra implementation has been developed as a supplementary exercise. This implementation incorporates an existing frontal face detection and simultaneously invokes the designed CNN in real-time for emotion classification. This added feature has now become a standard component in the assignments.

<p align="center" width="100%">
    <img src="images\let-s-get-emotional.gif" alt="Visualisation of the NEAT algorithm collecting data for training" width="70%">
</p>

## Implementation Details
The code is written in Python and relies on the packages described in [requirements.txt](requirements.txt). The most important packages used are:
* PyTorch
* OpenCV

For AI functionality, the code leverages the following techniques/algorithms:
* Convolutional Neural Networks
* Facial detection
* (Partly) Fine-tuning CNNs
* Designing and training CNNs
* Load balancers
* Data augmentation

## How to use
First install the requirements.txt.
```bash
pip install -r requirements.txt
```

Since the FER2013 dataset is quite large, the file isn't included on GitHub. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition). Place the `.csv` file in `data/FER`

Almost all architectures implemented in [nets.py](nets.py) can be used during training by running [main.py](main.py). To train a model yourself, run `python main.py -h` for input assisstance. An accepted example input is as follows:
```bash
python main.py -network JNet1 -loader small
```

The same applies to the script for fine-tuning the existing models, [finetuning.py](finetuning.py). For input assisstance, run `python finetuning.py -h`. An accepted example input is as follows:
```bash
python finetuning.py
```

You can also classify your own emotion using your webcam!
```bash
python camera.py
```