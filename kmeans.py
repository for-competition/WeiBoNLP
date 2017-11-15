from tf_idf_cal import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
#tf-idf features
dataset = features()
print(dataset)
print(dataset.shape)
#split dataset into train and test with 20 percent
train_x, test_x = train_test_split(dataset,test_size=0.2)
scores = []
for i in range(2,10):
    km = KMeans(n_clusters=i)
    km.fit(train_x)
    print(km.inertia_)  # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选择临界点的簇的个数
    scores.append(-km.score(test_x))
#
print(scores)
