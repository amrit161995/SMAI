import pandas as pd 
import math
import numpy as np
import statistics 
import pprint
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import sys

def data_preprocess(train_data_percent):
	data = pd.read_csv("AdmissionDataset/data.csv")
		# print(data)
	# data.loc[:,'GRE Score':'Research'] = (data.loc[:,'GRE Score':'Research']  -  data.loc[:,'GRE Score':'Research'].mean())/data.loc[:,'GRE Score':'Research'].std()
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

# def gradient_descent(train_data):
# 	Y = np.array(train_data['Chance of Admit '])
# 	# Y=Y.transpose()
# 	features_data=train_data.drop(['Chance of Admit '], axis=1)
# 	features_data=features_data.drop(['Serial No.'], axis=1)
# 	X = np.array(features_data)
# 	theta =  np.random.rand(8,)
# 	for k in range(2000):
# 		z=np.dot(X,theta)
# 		S = 1/(1 + np.exp(-z))
# 		difference =( S - Y)
# 		deri=np.dot(X.transpose(),difference)
# 		deri=0.1*(deri/len(X))
# 		theta=theta - deri/4
# 	return theta

def predict(coefficient,record,threshold):
	test=np.array(record)
	z=np.dot(test,coefficient)
	S = 1/(1 + np.exp(-z))
	if(S>threshold):
		return 1
	else:
		return 0
	

def validate_logistic(validate_data,coefficient,threshold):
	original_result = validate_data['Chance of Admit ']
	original_result=list(original_result)
	validate_data=validate_data.drop(['Chance of Admit '],axis=1)
	validate_data=validate_data.drop(['Serial No.'],axis=1)
	val=np.array(validate_data)
	predicted = []
	# print(original_result)
	for i in range(len(original_result)):
		if(original_result[i]>threshold):
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
		predicted_label = predict(coefficient,val[i],threshold)
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
	print("Results for Logistic Regression")
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

def eucidean_distance(x,y,attributes):
	dist_sq=0
	for i in attributes:
		dist_sq+=math.pow(x[i]-y[i],2)
	return math.sqrt(dist_sq)

def calc_distances(record,train_data,attribute_list,dist_formula):
	distances=[]
	train=np.array(train_data.values)
	for i in train:
		distances.append((dist_formula(record,i,attribute_list),i))
	return distances

def get_k_neighbours(distances,k):
	k_neighbours=[]
	distances.sort(key=lambda x: x[0])
	for i in range(k):
		k_neighbours.append(distances[i])
	return k_neighbours

def get_label(k_neighbours):
	label_list=[]
	for i in k_neighbours :
		label_list.append(i[1][-1])
	mappings={}
	unique_elements = np.unique(label_list)
	for i in unique_elements:
		mappings[i]=0

	for i in label_list:
		mappings[i]+=1
	count=0
	best_label=-1
	for key,value in mappings.items():
		if value >= count:
			count=value
			best_label=key
	return best_label

def validate_knn(train_data,validate_data,attribute_list,dist_formula,k):
	validate=np.array(validate_data.values)
	correct=0
	total=len(validate)
	false_positive=0
	false_negative=0
	true_positive=0
	true_negative=0
	
	# for calculation of accuracy
	for i in validate:
		distances=calc_distances(i,train_data,attribute_list,dist_formula)
		k_neighbours=get_k_neighbours(distances,k)
		predicted_label=get_label(k_neighbours)
		if(i[-1] == predicted_label):
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
	print("Results for KNN")
	print("True Positives = ",true_positive," True Negatives = ",true_negative)
	print ("False Positives = ",false_positive,"False Negatives = ",false_negative)
	Accuracy = correct/len(validate)
	Precision = true_positive/(true_positive+false_positive)
	Recall = true_positive/(true_positive+false_negative)
	F1 = 2/(1/Recall + 1/Precision)
	print("Accuracy = ",Accuracy)
	print("Precision = ",Precision)
	print("Recall = ",Recall)
	print("F1 Measure = ",F1)
	return Accuracy


def main():
	train_data,validate_data=data_preprocess(80)
	coefficient=coefficient_matrix_generation(train_data)
	validate_logistic(validate_data,coefficient,0.5)
	print()
	train_data['Chance of Admit '] = train_data['Chance of Admit '].apply(lambda x: 1 if x > 0.5 else 0)
	validate_data['Chance of Admit '] = validate_data['Chance of Admit '].apply(lambda x: 1 if x > 0.5 else 0)
	attribute_list=[1,2,3,4,5,6]
	knn_accuracy=validate_knn(train_data,validate_data,attribute_list,eucidean_distance,5)

main()