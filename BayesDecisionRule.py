import re
import pandas as pd
from math import log, ceil ,floor
import numpy as np
import random
from decimal import *
import copy
import time

global measurements
##############################################################################################################################################################################
################# Evaluate (Expected Gain, Accuracy and Confusion Matrix) ####################################################################################################

def Expected_Gain(Classes, economic_gain, confusion_matrix, total):
    expected_gain = 0
    for j in range(Classes):
        for k in range(Classes):
            expected_gain += (economic_gain[j][k]*confusion_matrix[j][k])/float(total)
    return round(expected_gain,3)

def Accuracy(confusion_matrix):
    print (confusion_matrix)
    acc = float(confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])  
    return round(acc,3)

def Confusion_Matrix(Classes,pred,actual):
    evaluations = []
    for i in range(Classes):
        evaluations.append([0 for k in range(Classes)])

    for i in range(len(pred)):
        evaluations[int(pred[i])][int(actual[i])]+=1
    #for i in range(Classes):
    #    for j in range(Classes):
    #        evaluations[i][j] /= float(total)
    return evaluations

##############################################################################################################################################################################
################# Calculate and Apply Decision Rule ##########################################################################################################################

def Apply_Rule(DATA, Memory, bins, measurements, num_levels):
    Predict = []
    for i in range(len(DATA)) :
        #instance = np.array(DATA.iloc[[i]])[0]
        instance = DATA[i]
        tuple_ = Bin_Finder(instance, bins, measurements)
        address = Tuple_to_Address(tuple_, measurements, num_levels)
        Predict.append(Memory[address][3])
    return Predict

def Cal_Decision_Rule(Memory, Classcounts, smoothingoffset, data_size, p0, p1):
    for i in range(len(Memory)):
        this  = copy.deepcopy(Memory[i])
        this[0]/= float(data_size+2*smoothingoffset)
        this[1]/= float(Classcounts[0]+smoothingoffset)
        this[2]/= float(Classcounts[1]+smoothingoffset)
        
        denom = (this[2]*p1)+(this[1]*p0)
        if((this[1]*p0)/denom > (this[2]*p1)/denom):
            Memory[i].append(0)
        else:
            Memory[i].append(1)
    return Memory

def Memory_Cells(DATA,bins,Memory, measurements, num_levels):
    ADDRESS = []
    bin_address = []
    for i in range(len(DATA)) :
        #instance = np.array(DATA.iloc[[i]])[0]
        instance = DATA[i]
        tuple_ =  Bin_Finder(instance, bins, measurements)
        address = Tuple_to_Address(tuple_, measurements, num_levels)
        ADDRESS.append(address)
        bin_address.append(Tostring(tuple_))
        Memory[address][0]+=1
        if(int(instance[5])==0):
            Memory[address][1]+=1
        else:
            Memory[address][2]+=1
    return Memory, ADDRESS, bin_address

def Bin_Finder(instance, bins, measurements):
    res = []
    for i in range(measurements):
        for j in range(len(bins[i])):
            if(float(bins[i][j]) > float(instance[i])):
                res.append(j-1)
                break
        if(instance[i]==bins[i][len(bins[i])-1]):
            res.append(len(bins[i])-2)
    return tuple(res)

def Initial_Bins(DATA,M,DATA_train, measurements):
    entropies = [float(i) for i in Entropies(DATA, measurements)]
    entropies = np.array(entropies)
    fj = [i/entropies.sum() for i in entropies]
   
    #number of bins for each dimensions
    K = [int(floor(M**i)) for i in fj]
    print ("#bins for each of dimensions",K)
    
    bins = []
    for i in range(measurements):
        sorted_unique_values = sorted(set(DATA_train.iloc[:,i]))
        N1 = len(sorted_unique_values)
        index = [int(round(j*(N1/K[i])+1)) for j in range(1, int(K[i]))]
    
        boarders = []
        boarders.append(float(0.0000))
        for j in index:
            bound = sorted_unique_values[j]
            boarders.append(round(bound,4))
        boarders.append(float(1.0000))
        #print boarders
        bins.append(boarders)
    return K, bins

def Entropies(DATA, measurements):
    print ("Entropies function")
    N = len(DATA)
    K = 10000
    entropies = []
    for i in range(measurements):
        chunks = {}
        entropy = 0
        for j in range(K+1):
            key = Decimal(float(j)/K).quantize(Decimal('.0001'))
            chunks[key] = 0
        #column = DATA.iloc[:,i]
        column = DATA[i]
        for item in column:
            item = Decimal(item).quantize(Decimal('.0001'))
            chunks[item] += 1
        zeros = 0
        for key in chunks.keys():
            if(chunks[key]==0):
                zeros +=1
            else:
                #calculate p_k
                p_k = float(chunks[key])/N
                #calculate entropy
                entropy += -1*p_k*log(p_k,2)
        entropy += (zeros-1)/(2*N*log(2))
        entropies.append(entropy)
    return entropies
def Tostring(tuple_):
    tuple_ = np.array(tuple_)
    res = ""
    for i in tuple_:
        res += str(i)
    return res

def Tuple_to_Address(tuple_, measurements, num_levels):
    res = tuple_[0]
    for j in range(1, measurements):
        res = res*num_levels[j] + tuple_[j]
    return res

##############################################################################################################################################################################
################# Load and Split Data ########################################################################################################################################

def Load_Dataset(address, portion, sizeofdata):
    #sizeofdata = 10000000
    data_size = int(sizeofdata*portion)
    measurements = 5
    #M = 10000
    #Reading dataset :
    Classes = 2
    Data = []
    counter = 0 
    with open(address) as data:
        for line in data:
            if(counter<data_size):
                counter+=1
                Data.append([float(i) for i in line.split()])
    return measurements, data_size, Classes, Data

def Split_data(data_size, Data):
    flag = [False for i in range(int(data_size))] 
    random.seed(2018)
    part1 = []
    one_third = int(data_size/3)
    counter = 0
    part2 = []
    while(counter!=one_third):
        index = random.randint(0,data_size-1)
        if (flag[index]== False):
            part1.append(Data[index])
            flag[index] = True
            counter+=1
    counter = 0
    part2 = []
    while(counter!=one_third):
        index = random.randint(0,data_size-1)
        if (flag[index]== False):
            part2.append(Data[index])
            flag[index] = True
            counter+=1
    part3 = []
    for i in range(len(flag)):
        if(flag[i]==False):
            part3.append(Data[i])
    return part1, part2, part3


##############################################################################################################################################################################
################# Calculate Rule and Evaluate It #############################################################################################################################

def All_in_One_Evaluate(part1, part2, DATA_train, DATA_dev, num_levels, bins, M, Memory, measurements, Classcounts, total_cells, data_size, p0, p1, Classes, economic_gain):
    
    Memory, ADDRESS, bin_address = Memory_Cells(part1, bins, Memory, measurements, num_levels)
    DATA_train["address"] = ADDRESS
    DATA_train["bin"] = bin_address
    
    Cal_Decision_Rule(Memory, Classcounts, total_cells, data_size, p0, p1)
    prediction = Apply_Rule(part2, Memory, bins, measurements, num_levels)

    actual = DATA_dev.loc[:,5]
    confusion_matrix = Confusion_Matrix(Classes,prediction,actual)
    accuracy = Accuracy(confusion_matrix)
    print "confusion_matrix", confusion_matrix
    expectedgain = Expected_Gain(Classes, economic_gain, confusion_matrix,len(DATA_dev))    
    return DATA_train, DATA_dev, accuracy, expectedgain , bins, Memory, actual
##############################################################################################################################################################################
################# Final Evaluation on Test ###################################################################################################################################

def Final_Evaluate(part3, DATA_test, num_levels, bins, M, Memory, measurements, Classcounts, total_cells, data_size, p0, p1, Classes, economic_gain):
 
    prediction = Apply_Rule(part3, Memory, bins, measurements, num_levels)

    actual = DATA_test.loc[:,5]
    confusion_matrix = Confusion_Matrix(Classes,prediction,actual)
    accuracy = Accuracy(confusion_matrix)

    expectedgain = Expected_Gain(Classes, economic_gain, confusion_matrix,len(DATA_test))    
    return accuracy, expectedgain

