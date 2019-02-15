import pandas as pd 
import math
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics
def data_preprocess():
	data = pd.read_csv("intrusion_detection/data.csv")
	# print (data)
	Y = data[['xAttack']]
	# print (Y)
	X = data.loc[:,'duration':'dst_host_srv_rerror_rate']
	X.loc[:,'duration':'dst_host_srv_rerror_rate'] = (X.loc[:,'duration':'dst_host_srv_rerror_rate']  -  X.loc[:,'duration':'dst_host_srv_rerror_rate'].mean())/X.loc[:,'duration':'dst_host_srv_rerror_rate'].std()
	# print(X)
	return X,Y

def covariance(X):
	features = X.transpose()
	cov_mat = np.cov(features)
	return cov_mat

def pca():
	x,y=data_preprocess()
	co=covariance(x)
	eig_val,eig_vec = np.linalg.eig(co)
	total=0
	d=0
	while total<0.9:
		total+=eig_val[d]/sum(eig_val)
		d+=1
	projected_x = x.dot(eig_vec.T[0:d].T)
	result = pd.DataFrame(projected_x)
	# print(result)
	# tr_re = result.dot(eig_vec.T[0:d])
	# for i in range(d,29):
	# 	result[i] = 0
	result['label'] = y
	return result

def hierarchical(k):
	result = pca()
	result = result[0:20000]
	Y = result[['label']]
	# print (Y)
	X = result.iloc[:,0:14]

	cluster = AgglomerativeClustering(n_clusters = k,affinity = 'euclidean' , linkage='ward')
	labels=cluster.fit_predict(X)
	Clusters = pd.DataFrame(result['label'])
	Clusters['Cluster'] = np.array(labels)
	return Clusters


def purity(Clusters,k):
	Cluster_purity = {}
	total = 0
	maxcount=0

	for i in range(k):
		print("For Cluster",i)
		cluster_data = Clusters[Clusters.Cluster == i ]
		# print(cluster_data['label'])
		uni,coun = np.unique(cluster_data['label'],return_counts=True)
		total+= sum(coun)
		maxcount+=max(coun)
		maxc=0
		max_label=''
		for j in range(len(coun)):
			if(coun[j]>maxc):
				maxc = coun[j]
				max_label = uni[j]
		Cluster_purity[i] = (max_label,maxc/sum(coun))
		print(uni)
		print(coun)
	return Cluster_purity,maxcount/total
	print(uni)
	print(coun)
def main():
	k=5
	Clusters=hierarchical(k)
	cluster_purity,total_purity=purity(Clusters,k)
	print ("Each Cluster Purity is given By :")
	for key,values in cluster_purity.items():
		print ("Cluster",key,"-------------------------------------------")
		print ("Cluster Label :",values[0])
		print ("Cluster Purity :",values[1])
		print()
	print("Total Purity",total_purity)


main()

# print (x.shape[0],len(x))
# err = np.sum(np.sum(np.square(np.array(tr_re) - np.array(x))))/len(x)
# print(err)

# print(x)
# print("=============================================")
# print (tr_re)
# print (eig_val[2]/sum(eig_val)) 
