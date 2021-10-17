import re
import pandas as pd
from math import log, ceil ,floor
import numpy as np
import random
from decimal import *
import copy
import time
import itertools
import BayesDecisionRule

def Volume (tuple_, bins):
	volume = 1
	for i in range(len(tuple_)):
		volume *= bins[i][tuple_[i]+1] - bins[i][tuple_[i]]
	return volume

def smoothed_Decision_Rule(Memory, Classcounts, smoothingoffset, b_m, data_size):
	for i in range(len(Memory)):
		this  = copy.deepcopy(Memory[i])
		this[0] += float(b_m[i][0])
		this[1] += float(b_m[i][1])
		this[2] += float(b_m[i][2])
		
		#print this , 3/float(Classcounts[0]+smoothingoffset)
		a= this[0]/float(data_size+2*smoothingoffset)
		b= this[1]/float(Classcounts[0]+smoothingoffset)
		c= this[2]/float(Classcounts[1]+smoothingoffset)
		denom = (c*p1)+(b*p0)
		
		#print (b*p0)/denom , (c*p1)/denom
		if((this[1]*p0)/denom > (this[2]*p1)/denom):
			Memory[i][3] = 0
		else:
			Memory[i][3] = 1
	return Memory

def caculate_alpha(Memory, VOL, Collective_vol, b_m):
	p_m = [0 for i in range(len(Memory))]
	denom = 0
	p_m_start = [0 for i in range(len(Memory))]
	for i in range(len(Memory)):
		b_m_ = b_m[i][0]
		v = VOL[i]
		v_m = float(Collective_vol[i])
		this_p_m = (b_m_ * v)/ (v_m)
		p_m_start[i] = this_p_m
		denom += this_p_m
	alpha = 1/denom
	p_m = np.array(p_m_start)*alpha
	#p_m = [round(i,5) for i in range(len(p_m))]
	#print p_m_start[0], alpha, p_m[0]
	return alpha, p_m

def Evaluate(k,b_m,Memory, total_cells, actual, bins, DATA_dev, Classcounts, measurements, num_levels, part1, part2, economic_gain, data_size):
	#Caculating maximum liklihood probabilty
	#print economic_gain, type(economic_gain)
	Classes = 2
	#p_m = np.array(p_m)
	#P_m = alpha * p_m
	print("Ready")   
	new_Memory = np.array(Memory, copy=True) 
	new_Memory = smoothed_Decision_Rule(new_Memory, Classcounts,total_cells, b_m, data_size)
	prediction = BayesDecisionRule.Apply_Rule(part2, new_Memory, bins, measurements, num_levels)
	new_confusion_matrix = BayesDecisionRule.Confusion_Matrix(Classes,prediction, actual, )
	new_expected_gain = BayesDecisionRule.Expected_Gain(Classes, economic_gain, new_confusion_matrix, len(DATA_dev))
	new_accuracy = float(new_confusion_matrix[0][0]+new_confusion_matrix[1][1])/(new_confusion_matrix[0][0]+new_confusion_matrix[1][1]+new_confusion_matrix[0][1]+new_confusion_matrix[1][0])
	new_expected_gain = round(new_expected_gain,4)
	new_accuracy = round(new_accuracy,4)
	e = open('main.txt', 'a')
	e.write("new Acc")
	e.write(str(new_accuracy))
	e.write(", nex_Ex")
	e.write(str(new_expected_gain))
	e.write("k")
	e.write(str(k))
	e.write("\n")
	print ("new Acc",new_accuracy,", nex_Ex",new_expected_gain, "k", k)
	e.close()
	return
'''
def Tuple_to_Address(tuple_,measurements,num_levels):
	res = tuple_[0]
	for j in range(1, measurements):
		res = res*num_levels[j] + tuple_[j]
	return res'''
############################################################################################################
def initialize(leng, bins, measurements, num_levels):
	VOL = [0 for i in range(leng)]
	x = [0,1,2,3,4,5]
	P = [k for k in itertools.product(x,repeat=5)]
	for permutaion in P:
		address = BayesDecisionRule.Tuple_to_Address(permutaion, measurements, num_levels)
		v = Volume(permutaion, bins)
		VOL[address] = v
	return VOL,P
############################################################################################################
def Smoothing(part1, part2, Memory, bins, DATA_train, DATA_dev, measurements, num_levels, total_cells, actual, Classcounts, economic_gain, data_size):
	global p0,p1
	#print Memory
	p0, p1 = 0.4, 0.6
	leng = len(Memory)
	VOL, P = initialize(leng, bins, measurements, num_levels)
	N = len(DATA_train)
	M = len(Memory)
	J = [1,2,3,4]
	#k = [round(float(j*N)/(20*M)) for j in J]
	k   = [round(float(j*N)/(20*M)) for j in J]
	b_m = [[[0,0,0] for i in range(len(P))] for j in range(len(J))]
	print "________k",k
	Collective_vol = [[0 for i in range(len(Memory))] for j in J]
	for permutation in P :
		permutation = np.array(permutation)
		address = BayesDecisionRule.Tuple_to_Address(permutation, measurements, num_levels)
		#density = Memory[address][0]
		#pos_dens = Memory[address][1]
		#neg_dens = Memory[address][2]
		count, volume = [[0,0,0] for j in J], [0 for j in J]
		level = 1
		Flag = [False for j in J]
		for j in range(len(J)):
				volume[j] = VOL[address]
		while(count[len(count)-1][0]<k[len(count)-1]):
			if(level>5):
					break
			for i in range(measurements):
				if(permutation[i]-level>=0):
					neighbour = permutation
					neighbour[i] -= level
					neighbour_address = BayesDecisionRule.Tuple_to_Address(neighbour, measurements, num_levels)
					for j in range(len(J)):
						if(not Flag[j]):
							count[j][0] += Memory[neighbour_address][0]
							count[j][1] += Memory[neighbour_address][1]
							count[j][2] += Memory[neighbour_address][2]
							#pos_dens += Memory[neighbour][1]
							#neg_dens += Memory[neighbour][2]
							volume[j]   += VOL[neighbour_address]
							if(count[j][0]>=k[j]):
								Flag[j] = True
				if(permutation[i]+level<=5):
					neighbour = permutation
					neighbour[i] += level
					neighbour_address = BayesDecisionRule.Tuple_to_Address(neighbour, measurements, num_levels)
					for j in range(len(J)):
						if(not Flag[j]):
							count[j][0] += Memory[neighbour_address][0]
							count[j][1] += Memory[neighbour_address][1]
							count[j][2] += Memory[neighbour_address][2]
							volume[j] += VOL[neighbour_address]
							if(count[j][0]>=k[j]):
								Flag[j] = True
			level += 1
		for j in range(len(J)):
			b_m[j][address][0] = count[j][0]
			b_m[j][address][1] = count[j][1]
			b_m[j][address][2] = count[j][2]
			Collective_vol[j][address] = volume[j]
			#print b_m[j][address], count[j]
	#for j in range(len(J)):
	#	print b_m[j]
	for j in range(len(J)):
		#alpha, p_m = caculate_alpha(Memory,VOL, Collective_vol[j], b_m[j])
		Evaluate(k[j],b_m[j], Memory, total_cells, actual,bins, DATA_dev, Classcounts, measurements, num_levels, part1, part2, economic_gain, data_size)
	return
