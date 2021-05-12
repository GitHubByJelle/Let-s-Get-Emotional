import numpy as np
import os

import loaders

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

trainloader, valloader, testloader, plotloader = loaders.make_transformer_loader(25)
loader = trainloader

n,m = 5,5
fig, ax = plt.subplots(n,m)

# Get data
dataiter = iter(loader)
data, label = dataiter.next()

for i in range(n):
    for j in range(m):
        image = data[i*m+j][0]
        ax[i,j].imshow(image, cmap='gray')
        ax[i,j].axis('off')
        ax[i,j].title.set_text(classes[int(label[i*m+j])])
plt.show()

