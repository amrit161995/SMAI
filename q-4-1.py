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
	data = pd.read_csv("AdmissionDataset/data.csv")
	data.loc[:,'GRE Score':'Research'] = (data.loc[:,'GRE Score':'Research']  -  data.loc[:,'GRE Score':'Research'].mean())/data.loc[:,'GRE Score':'Research'].std()
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

def coefficient_matrix_generation(train_data):
	Y = np.array(train_data['Chance of Admit '])
	features_data=train_data.drop(['Chance of Admit '], axis=1)
	features_data=features_data.drop(['Serial No.'], axis=1)
	X = np.array(features_data)
	transpose_x = X.transpose()
	y= Y.transpose()
	a=np.matmul(transpose_x,X)
	b=np.linalg.inv(a) 
	c=np.matmul(b,transpose_x)
	coefficient = np.matmul(c,Y)
	return coefficient

def gradient_descent(train_data):
	Y = np.array(train_data['Chance of Admit '])
	# Y=Y.transpose()
	features_data=train_data.drop(['Chance of Admit '], axis=1)
	features_data=features_data.drop(['Serial No.'], axis=1)
	X = np.array(features_data)
	theta =  np.random.rand(8,)
	for k in range(2000):
		z=np.dot(X,theta)
		S = 1/(1 + np.exp(-z))
		difference =( S - Y)
		deri=np.dot(X.transpose(),difference)
		deri=0.1*(deri/len(X))
		theta=theta - deri/4
	return theta

def predict(coefficient,record):
	test=np.array(record)
	z=np.dot(test,coefficient)
	S = 1/(1 + np.exp(-z))
	if(S>0.5):
		return 1
	else:
		return 0
	

def validate_logistic(validate_data,coefficient):
	original_result = validate_data['Chance of Admit ']
	original_result=list(original_result)
	validate_data=validate_data.drop(['Chance of Admit '],axis=1)
	validate_data=validate_data.drop(['Serial No.'],axis=1)
	val=np.array(validate_data)
	predicted = []
	# print(original_result)
	for i in range(len(original_result)):
		if(original_result[i]>0.5):
			original_result[i]=1
		else:
			original_result[i]=0
	# print(original_result)
	
	correct=0
	total=len(val)
	false_positive=0
	false_negative=0
	true_positive=0
	true_negative=0
	
	# for calculation of accuracy
	for i in range(len(val)):
		predicted_label = predict(coefficient,val[i])
		if ( predicted_label== original_result[i]):
			correct+=1
			if (predicted_label == 1):
				true_positive+=1
			else:
				true_negative+=1
		else:
			if (predicted_label == 1):
				false_positive+=1
			else:
				false_negative+=1
	print("True Positives = ",true_positive," True Negatives = ",true_negative)
	print ("False Positives = ",false_positive,"False Negatives = ",false_negative)
	Accuracy = correct/len(val)
	Precision = true_positive/(true_positive+false_positive)
	Recall = true_positive/(true_positive+false_negative)
	F1 = 2/(1/Recall + 1/Precision)
	print("Accuracy = ",Accuracy)
	print("Precision = ",Precision)
	print("Recall = ",Recall)
	print("F1 Measure = ",F1)
	return Accuracy


def r2(y_true,y_pred):
	return r2_score(y_true, y_pred)  

def main():
	train_data,validate_data=data_preprocess(80)
	coefficient=gradient_descent(train_data)
	validate_logistic(validate_data,coefficient)
	

main()