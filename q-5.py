import pandas as pd 
import math
import numpy as np
import statistics 
import pprint
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score
import sys
def data_preprocess(train_data_percent):
	data = pd.read_csv("wine-quality/data.csv",header=None)
	# header =data.iloc[0].str.split(';',expand=True)
	data=data.drop(data.index[0])
	data[[1,2,3,4,5,6,7,8,9,10,11,12]] = (data[0].str.split(';',expand=True))
	data=data.drop([0],axis=1)

	data= data.astype('float64') 
	data.loc[:,1:11] = (data.loc[:,1:11]  -  data.loc[:,1:11].mean())/data.loc[:,1:11].std()
	
	data=data.sample(frac=1)
	
	total_length = len(data)
	padding = []
	for i in range(total_length):
		padding.append(1)
	data.insert(0,0,padding)
	Train_data_count=math.floor((train_data_percent/100)*len(data.values))
	Test_data_count=len(data.values)-Train_data_count
	train_data=data[0:Train_data_count]
	validate_data=data[Train_data_count:]
	return train_data,validate_data

def coefficient_matrix_generation(X,Y):
	transpose_x = X.transpose()
	y= Y.transpose()
	a=np.dot(transpose_x,X)
	b=np.linalg.inv(a) 
	c=np.matmul(b,transpose_x)
	coefficient = np.matmul(c,Y)
	return coefficient

def gradient_descent(X,Y):
	# Y = np.array(train_data['Chance of Admit '])
	# # Y=Y.transpose()
	# features_data=train_data.drop(['Chance of Admit '], axis=1)
	# features_data=features_data.drop(['Serial No.'], axis=1)
	# X = np.array(features_data)
	theta =  np.random.rand(12,)
	for k in range(500):
		z=np.dot(X,theta)
		S = 1/(1 + np.exp(-z))
		difference =( S - Y)
		deri=np.dot(X.transpose(),difference)
		deri=0.1*(deri/len(X))
		theta=theta - deri/2
	return theta

def one_vs_all(train_data,validate_data):
	theta_dict={}
	for i in range(3,10):
		Y = np.array(train_data[12])
		features_data=train_data.drop([12], axis=1)
		X = np.array(features_data)
		for j in range(len(Y)):
			if(i == Y[j]):
				Y[j] = 1
			else:
				Y[j] = 0
		theta = gradient_descent(X,Y)
		theta_dict[i]=theta
	print("One Vs All")
	validate(validate_data,theta_dict,"one_vs_all")
		# validate()

def one_vs_one(train_data,validate_data):
	theta_dict={}
	for i in range(3,10):
		for j in range(i+1,10):
			Y = np.array(train_data[12])
			features_data=train_data.drop([12], axis=1)
			X = np.array(features_data)
			new_X=[]
			new_Y=[]
			for k in range(len(Y)):
				if(i == Y[k]):
					new_X.append(X[k])
					new_Y.append(1)
				elif(j == Y[k]):
					
					new_X.append(X[k])
					new_Y.append(0)
			theta = gradient_descent(np.array(new_X),np.array(new_Y))
			theta_dict[(i,j)]=theta
	print("One Vs One")
	validate(validate_data,theta_dict,"one_vs_one")



def predict(coefficient,record,type1):
	test=np.array(record)
	sigmoid_values = {}
	for key,values in coefficient.items():
		z=np.dot(test,values)
		S = 1/(1 + np.exp(-z))
		sigmoid_values[key] = S
	max_key = 0
	max_value = 0
	for key,values in sigmoid_values.items():
		if values>=max_value:
			max_value=values
			max_key = key
	if type1 == 'one_vs_all':
		return max_key
	
	if (max_value>=0.5):
		return max_key[0]
	else:
		return max_key[1]

def validate(validate_data,coefficient,type1):
	original_result = validate_data[12]
	original_result=list(original_result)
	validate_data=validate_data.drop([12],axis=1)
	val=np.array(validate_data)
	
	correct=0
	total=len(val)
	
	# for calculation of accuracy
	for i in range(len(val)):
		predicted_label = predict(coefficient,val[i],type1)
		if ( predicted_label == original_result[i]):
			correct+=1
			
	Accuracy = correct/len(val)
	print("Accuracy = ",Accuracy)
	return Accuracy


def main():
	train_data,validate_data=data_preprocess(80)
	one_vs_all(train_data,validate_data)
	one_vs_one(train_data,validate_data)
	# coefficient=coefficient_matrix_generation(train_data)
	# print(coefficient)
	# # validate_logistic(validate_data,coefficient)
	

main()