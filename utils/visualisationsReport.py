import pandas as pd
import matplotlib.pyplot as plt

# Plot for distribution emotions in dataset
class_dict = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'sadness',5:'surprise',6:'neutral'}
df = pd.read_csv('data/FER/fer2013.csv')
df_count = df.groupby('emotion').count()
labels = [class_dict[x] for x in df_count.index.tolist()]
values = df_count.pixels.tolist()
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=-20)
plt.axis('equal')
plt.show()

# Get number of train, test and validate images
print(df.groupby('Usage').count())
