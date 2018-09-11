from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt 

def get_cluster_centers(images, kList):
	retList = []
	for k in kList:
		print('on this k:', k)
		oldMean = images[np.random.choice(images.shape[0], k, replace=False),:]
		trueMean = images[np.random.choice(images.shape[0], k, replace=False),:]
		while not same_mean(trueMean, oldMean):
			oldMean = trueMean
			clusters = cluster_images(images, trueMean)
			trueMean = find_new_mean(oldMean, clusters)
		retList.append((trueMean, clusters))
	return retList

def cluster_images(images, mean):
	clusterDict  = {}
	for x in images:
		temp = []
		for i in enumerate(mean):
			norm = np.linalg.norm(x-mean[i[0]])
			temp.append((i[0], norm))
		best = min(temp, key=first_index)[0]
		try:
			clusterDict[best].append(x)
		except KeyError:
			clusterDict[best] = [x]
	return clusterDict

def same_mean(trueMean, oldMean):
	oldList, newList = [tuple(i) for i in oldMean], [tuple(i) for i in trueMean]
	oldMeanSet, trueMeanSet = set(oldList), set(newList)
	return oldMeanSet == trueMeanSet	

def find_new_mean(oldMean, clusters):
    clusterKeys = sorted(clusters.keys())
    newMean = []
    for k in clusterKeys:
        newMean.append(np.mean(clusters[k], axis=0, dtype=None, out=None, keepdims=False))
    return newMean

def first_index(x):
	return x[1]

def show_clusters():
	retList = get_cluster_centers(images, kList)
	for each in retList:
		mean = each[0]
		for i,m in enumerate(mean):
			plt.subplot(5,4,i+1)
			plt.imshow(m.reshape(28, 28), cmap='gray')
			plt.axis('off')
		plt.show()

mndata = MNIST('files')
images_all, _ = mndata.load_training()
images = np.array(images_all[:100])
kList = [5, 10, 20]
show_clusters()
