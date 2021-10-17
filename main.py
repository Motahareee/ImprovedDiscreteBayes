import re
import pandas as pd
from math import log, ceil ,floor
import numpy as np
import random
from decimal import *
import copy
import time
import Optimization
import BayesDecisionRule
import smoothing



#read dataset and split it to train, develop and test set
address = "/Users/Moti/Desktop/pr_data.txt"

#Portion = [1]#0.00001]#,0.1,1]#[0.0001,0.001,0.01,0.1, 1]
Portion = [0.01,0.1,1]#0.00001]#,0.1,1]#[0.0001,0.001,0.01,0.1, 1]

global num_levels
num_levels = [7,7,7,7,7,7]
bins, M = [], 10000

#initialize memorycells
total_cells = 1
for i in num_levels:
    total_cells *= i
#Memory = [[2,1,1] for i in range(total_cells)]

global p0,p1,Classcounts,economic_gain #accuracy, expectedgain
p0, p1 = 0.4, 0.6
economic_gain = [[1, -1],[-2, 3]]
sizeofdata = 10000000


measurements, data_size, Classes, Data = BayesDecisionRule.Load_Dataset(address,1, sizeofdata)
PART1, PART2, PART3 = BayesDecisionRule.Split_data(data_size, Data)
DATA_test  = pd.DataFrame(PART3)
#initializer
num_levels, bins = BayesDecisionRule.Initial_Bins(PART1,M,pd.DataFrame(PART1), measurements)
e = open('main.txt', 'a')
e.write("Equal Probability Quantization--------------------------------------------------------\n")
e.write(str(bins))
e.write("--------------------------------------------------------------------------------------\n")
bins = [[0.0, 0.01999, 0.17985, 0.184, 0.564, 0.90893, 1.0], [0.0, 0.47461, 0.49098, 0.53597, 0.59096, 0.62, 1.0], [0.0, 0.08492, 0.15096, 0.24194, 0.60667, 0.94591, 1.0], [0.0, 0.22481, 0.25479, 0.32474, 0.326, 0.852, 1.0], [0.0, 0.064, 0.19, 0.51475, 0.516, 0.602, 1.0]]
seed = 20185

for portion in Portion:
    bound =  int(portion*len(PART1))
    part1, part2 = PART1[0:bound], PART2[0:bound] 
    DATA_train = pd.DataFrame(part1)
    DATA_dev   = pd.DataFrame(part2)
    Classcounts = DATA_train.iloc[:,5].value_counts()


    Memory = [[2,1,1] for i in range(total_cells)]
    DATA_train, DATA_dev, accuracy, expectedgain, bins, Memory, actual = BayesDecisionRule.All_in_One_Evaluate(part1, part2, DATA_train, DATA_dev, num_levels, bins, M, Memory, measurements, Classcounts, total_cells, portion, p0, p1, Classes, economic_gain)
    #print "::::",len(DATA_train)
    accuracy = round(accuracy,4)
    expected_gain = round(expectedgain,4)
    Flag = True 
    while(Flag!=False):
        expectedgain, accuracy,bins, Memory, DATA_train = Optimization.Random_Optimize_Quantizers(part1, part2, DATA_train, DATA_dev, Memory,total_cells, accuracy, expectedgain, bins, measurements, seed, num_levels, data_size, Classcounts,p0,p1, Classes, actual, economic_gain, portion)
        seed +=1
        Flag, expectedgain, accuracy,bins, Memory, DATA_train = Optimization.InOrder_Optimize_Quantizers(part1, part2, DATA_train, DATA_dev, Memory,total_cells, accuracy, expectedgain, bins, num_levels, data_size, Classcounts,p0,p1, Classes, actual, economic_gain, measurements, portion)

e.write("BEST----------------------------------------------------------------------------------\n")
e.write(str(bins))
e.write("--------------------------------------------------------------------------------------\n")
#find final results

acc, exp =BayesDecisionRule.Final_Evaluate(PART3, DATA_test, num_levels, bins, M, Memory, measurements, Classcounts, total_cells, data_size, p0, p1, Classes, economic_gain)
print ("FINAL RESULT" ,acc, exp)
e.write("FINAL RESULT   Accuracy: " )
e.write(str(acc))
e.write("  Expected Gain: ")
e.write(str(exp))
e.write("\n")
e.close()
smoothing.Smoothing(PART1, PART2, Memory, bins, DATA_train, DATA_dev, measurements, num_levels, total_cells, actual, Classcounts, economic_gain, data_size)
