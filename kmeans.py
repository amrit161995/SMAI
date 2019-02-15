import pandas as pd 
import math
import numpy as np
import sys
from copy import deepcopy
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
	return result,d

# data = dimensionality_reduction()
# print (x.shape[0],len(x))

def eucidean_distance(x,y,start,end):
	dist_sq=[]
	x=np.array(x)
	for i in range(start,end+1):
		dist_sq+=math.pow(x[i]-y[i],2)
	return math.sqrt(dist_sq)


# def assignment(data,centroids):
# 	np_data = np.array(data)
# 	closest = [0]*len(np_data)
# 	for j in range(len(np_data)):
# 		mini = 1
# 		min_dist = 99999
# 		for i in centroids.keys():
# 			dist=eucidean_distance(np_data[j],centroids[i],0,22)
# 			if(dist<min_dist):
# 				mini = i
# 				min_dist = dist

# 		closest[j] = mini 
# 	return closest
	
def assignment(data,centroids,dimensions):
	# print (data)
	for i in centroids.keys():
		data['Cluster_' + str(i)]=0
		for j in range(0,dimensions):
			data ['Cluster_'+str(i)] +=(data[j] - centroids[i][j])**2
		data['Cluster_' + str(i)] = np.sqrt(data ['Cluster_'+str(i)])
	dist_cols = ['Cluster_'+str(i) for i in centroids.keys()]
	data['closest'] = data.loc[:,dist_cols].idxmin(axis=1)
	return data

def update(data,centroids,dimensions):
	for i in centroids.keys():
		mean=np.array(np.mean((data[data.closest == 'Cluster_'+str(i)].iloc[:,0:dimensions])))
		centroids[i]=mean
		# print (temp)
		# print("----------------------------------------")
	return centroids


def k_means(k):
	centroids = {}
	for i in range(1,k+1):
		centroids[i]=np.random.rand(14,)
	data,dimensions = pca()
		
	for i in range(20):
		data=assignment(data,centroids,dimensions)
		prev = deepcopy(centroids)
		centroids=update(data,centroids,dimensions)
		if(prev.values() == centroids.values()):
			break
		print(centroids)
	return data,centroids
	# print(useful)

def purity(data,centroids):
	Cluster_purity = {}
	total = 0
	maxcount=0
	for i in centroids.keys():
		print("For Cluster",i)
		temp = data[data.closest == 'Cluster_'+str(i)]['label']
		uni,coun=np.unique(temp,return_counts=True)
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
	


def main():
	data,centroids=k_means(5)
	cluster_purity,total_purity=purity(data,centroids)
	print ("Each Cluster Purity is given By :")
	for key,values in cluster_purity.items():
		print ("Cluster",key,"-------------------------------------------")
		print ("Cluster Label :",values[0])
		print ("Cluster Purity :",values[1])
		print()
	print("Total Purity",total_purity)
		
main()
# purity(uni,coun)
# print (c)
# print(x)
# print("=============================================")
# print (tr_re)
# print (eig_val[12]/sum(eig_val)) 
