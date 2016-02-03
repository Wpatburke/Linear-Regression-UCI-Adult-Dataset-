# William Burke
# Linear Regression
# January 22nd


#declare imports
import sys
import numpy as np
import time

# Time used to measure length of runtime 
#StartTime = time.time()
Lambda = 1

# raw datasets used
train_file = 'a7a.train' #sys.argv[1]
dev_file = 'a7a.dev'
file_test =sys.argv[1]

#open files into readable format
Training_Data = open(train_file, 'r')
Dev_Data = open(dev_file,'r')
Testing_Data = open(file_test,'r')

# Function to convert Raw data (the ones provided) into a useable Matrix X (dataMatrix), with dummy variables
def RawToX(infile):
	TrueValues = []
	dataMatrix = np.zeros(124) #declare our data matrix so we can append rows to it
	Row = infile.readline() 
	i = 0
	while Row!="": #until file empty
		# if i == 5:
		# 	print dataMatrix
		# 	break

		newRow = np.zeros(124)
		newRow[0] = 1

		variables = Row.split(" ")
		TrueValues.append(int(variables[0])) #take first value in each row as y
		#print variables
		#variables.pop(0)
		variables.pop(0) # remove so y value is not added to data matrix
		for value in variables:
			if value =='\n':
				break
			num = value.replace(":1","") #delete extraneous information
			#print num
			num = int(num)
			newRow[num] = 1

		dataMatrix = np.vstack([dataMatrix,newRow]) #append row to our data matrix
		Row = infile.readline()
		i = i + 1 
	#print len(TrueValues)
	dataMatrix = np.delete(dataMatrix,0,axis=0)  # delete the first row, it was just used to instantiate the matrix and is all zeros
	#print dataMatrix.shape
	#print TrueValues[6]
	#print dataMatrix[6]
	return dataMatrix, TrueValues


#used no regularization initially in codes first rough draft
def Model_No_Regularization(X,y):
	Beta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y) # perform equation
	yHat = np.dot(X,Beta) # calculate predictions with X and Beta

	yhatGuess = [] 

	for a in yHat:  # force each prediction to 1 or -1 by sign
		if a > 0:
			yhatGuess.append(1)
		else:
			yhatGuess.append(-1)
	return yhatGuess, Beta

def Model_Ridge_Regularization(X,y,Lambda):
	Beta = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(np.transpose(X),X), Lambda*np.identity(124,dtype=int))),np.transpose(X)),y) # Find weights
	yHat = np.dot(X,Beta) # calculate predictions with X and Beta

	yhatGuess = []

	for a in yHat:  # force each prediction to 1 or -1 by sign
		if a > 0:
			yhatGuess.append(1)
		else:
			yhatGuess.append(-1)
	return yhatGuess, Beta



def Model_Lasso_Regularization(X,y,Lambda):
	Beta = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(np.transpose(X),X), Lambda*np.identity(124,dtype=int))),np.transpose(X)),y) # Find weights
	#Beta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y) # Find weights
	
	#Lasso formulation
	# b = -1
	# while b < len(Beta)-1:
	# 	b = b + 1
	# 	if Beta[b] ==0:
	# 		continue
	# 	Beta[b] = (Beta[b]/float(abs(Beta[b])))*max((abs(Beta[b])-Lambda),0)
		
		#b = (b/abs(b))*max((abs(b)-Lambda),0)

	yHat = np.dot(X,Beta) # calculate predictions with X and Beta

	yhatGuess = []

	for a in yHat:  # force each prediction to 1 or -1 by sign
		if a > 0:
			yhatGuess.append(1)
		else:
			yhatGuess.append(-1)
	return yhatGuess, Beta
# function to evaluate how well my model performed
def Evaluate_Model(yHat,y): 

	Correct = 0
	positive = 0
	i = 0
	while i < len(yHat):
		if y[i] ==1:
			positive = positive + 1
		if y[i] == yHat[i]: # does our guess match the original?
			Correct = Correct + 1
		i = i + 1

	# print ("")
	# print Correct," correct predictions for ",i," points."
	# print "The accuracy is",Correct/float(i),"."
	# print ("")

	#print "psotive,", positive/float(i)
	#print "Correct: ", Correct
	#print "Incorrect: ", i - Correct
	#print "Total: ", i
	#print "Accuracy: ", Correct/float(i)

def Test_Model(X,y,Beta):
	yHat = np.dot(X,Beta)

	yhatGuess = []
	yAllPos = []
	for a in yHat: 
		yAllPos.append(1)
		if a > 0:
			yhatGuess.append(1)
		else:
			yhatGuess.append(-1)


	Correct = 0
	pos = 0
	pos1 = 0
	i = 0
	#print yHat


	while i < len(yhatGuess): 
		if y[i] == yhatGuess[i]:
			Correct = Correct + 1
		if y[i] == 1:
			pos = pos + 1
		if yhatGuess[i] == 1:
			pos1 = pos1 + 1

		i = i + 1

	print ("")
	print Correct," correct predictions for ",i," points."
	print "The accuracy is",Correct/float(i),"."
	print ("")

	#Additional debugging output: 
	#print "Predicted positive: ", pos1
	#print "Actually positive: ", pos
	#print y
	#print yHat
	#print yhatGuess
	#print("")

#function used to determine the best lambda to use from the dev data set
def findLambda(X,y,Lambda):

	Beta = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(np.transpose(X),X), Lambda*np.identity(124,dtype=int))),np.transpose(X)),y) # Find weights
	#Beta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y) # Find weights
	
	#Attempted Lasso formulation
	#Lasso formulation
	# b = -1
	# while b < len(Beta)-1:
	# 	b = b + 1
	# 	if Beta[b] == 0:
	# 		continue
	# 	Beta[b] = (Beta[b]/float(abs(Beta[b])))*max((abs(Beta[b])-Lambda),0)
		
	# Beta = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(np.transpose(X),X), Lambda*np.identity(124,dtype=float))),np.transpose(X)),y)
	yHat = np.dot(X,Beta)

	yhatGuess = []

	for a in yHat: 
		if a > 0:
			yhatGuess.append(1)
		else:
			yhatGuess.append(-1)


	Correct = 0
	i = 0
	while i < len(yHat):
		if y[i] == yhatGuess[i]:
			Correct = Correct + 1
		i = i + 1

	# print ("")
	# print Correct," correct predictions for ",i," points."
	# print "The accuracy is",Correct/float(i),"."
	# print ("")
	# return Correct/float(i) # returns accuracy


	#return yhatGuess, Beta


# First we need to find the lambda our model should use when finding weights.
	# I increment lambda from 0 to 100 by 0.1 and 
	# determine LambdaMax, the lambda with the max accuraxy

dataMatrix, TrueValues = RawToX(Dev_Data)
X = dataMatrix
y = TrueValues
AccuracyMax = 0
LambdaMax = 0


#UNCOMMENT OUT CODE TO RUN THE LAMBDA FINDER
################################################################################
# lambdas = [x * 0.1 for x in range(0, 1000)]
# for i in lambdas:
# 	#print "Lambda: ", i
# 	CurrentAccuracy = findLambda(X,y,i)
# 	#print "CurrentAccuracy: ", CurrentAccuracy
# 	if CurrentAccuracy > AccuracyMax:
# 		AccuracyMax = CurrentAccuracy 
# 		LambdaMax = i
#print "Largest Accuracy: ",AccuracyMax
#print "With LambdaMax: ", LambdaMax
################################################################################


LambdaMax = 0.7
# Secondly, I use the LambdaMax found to find the weights
dataMatrix, TrueValues = RawToX(Training_Data)

X = dataMatrix
y = TrueValues

#print "train"
yHat, Beta = Model_Ridge_Regularization(X,y,LambdaMax)
Evaluate_Model(yHat,y)
#print "test"
# Third and lastly, I use the lambda and weights to form predictions for the testing data set
X,y = RawToX(Testing_Data)
Test_Model(X,y,Beta)


print("=============================================================================")
print("=============================================================================")
