from tf_idf_cal import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
#tf-idf features
dataset = features()
print dataset.shape
#split dataset into train and test with 20 percent
train_x, test_x = train_test_split(dataset,test_size=0.2)
scores = []
for i  in range(2,10):
    km  = KMeans(n_clusters=i)
    km.fit(train_x)
    scores.append(-km.score(test_x))

print scores