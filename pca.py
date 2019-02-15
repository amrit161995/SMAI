import pandas as pd 
import math
import numpy as np
import statistics 
import pprint
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score
import sys

def data_preprocess():
	data = pd.read_csv("intrusion_detection/data.csv")
	Y = data[['xAttack']]
	X = data.loc[:,'duration':'dst_host_srv_rerror_rate']
	X.loc[:,'duration':'dst_host_srv_rerror_rate'] = (X.loc[:,'duration':'dst_host_srv_rerror_rate']  -  X.loc[:,'duration':'dst_host_srv_rerror_rate'].mean())/X.loc[:,'duration':'dst_host_srv_rerror_rate'].std()
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
	result['label'] = y
	return result

print (pca())
