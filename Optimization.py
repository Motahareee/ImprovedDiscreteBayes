import re
import pandas as pd
from math import log, ceil ,floor
import numpy as np
import random
from decimal import *
import copy
import time
import BayesDecisionRule

def compair(bins, new_bins):
    for i in range(len(bins)):
        for j in range(len(bins[0])):
            if(bins[i][j]!=new_bins[i][j]):
                print (bins[i][j], new_bins[i][j])
    return

def Update_Decision_Rule(change_memory_cells, new_Memory, smoothingoffset, data_size, Classcounts, p0, p1):
    for i in change_memory_cells:
        this  = [k for k in new_Memory[i]]
        this[0]/= float(data_size+2*smoothingoffset)
        this[1]/= float(Classcounts[0]+smoothingoffset)
        this[2]/= float(Classcounts[1]+smoothingoffset)
        #print "this",this
        denom = (this[2]*p1)+(this[1]*p0)
        #print "this____",this[1]*p0)/denom, (this[2]*p1)/denom
        if((this[1]*p0)/denom > (this[2]*p1)/denom):
            new_Memory[i][3] = 0
        else:
            new_Memory[i][3] = 1
    return new_Memory

def Update_Memory(dimension, b_new ,b_old, new_Memory, DATA_train, measurements, num_levels):
    change_memory_cells = []
    #new_DATA_train = DATA_train.copy()
    change_address = []
    if(b_old < b_new):
        change_rows = DATA_train.loc[(b_old<=DATA_train[dimension]) & (DATA_train[dimension]<b_new)]
        change_rows = change_rows.reset_index()
        #change_rows.columns = ['index', 0,1,2,3,4,5,'address','bin']
        change_rows = np.array(change_rows)
        for i in range(len(change_rows)):
            #print "HALA"
            former = int(change_rows[i][7])
            label = int(change_rows[i][6])
            new_Memory[former][0] -= 1
            new_Memory[former][label+1] -= 1
            tuple_ = [int(char) for char in change_rows[i][8]]
            tuple_[dimension] -= 1
            if(tuple_[dimension]==-1):
                print ("error", tuple_)
                print (b_new, b_old, dimension)
                print (DATA_train)
                return "FALSE"
            new = BayesDecisionRule.Tuple_to_Address(tuple_, measurements, num_levels)
            
            new_Memory[new][0] += 1
            new_Memory[new][label+1] += 1
            
            change_memory_cells.append(former)
            change_memory_cells.append(new)
            change_address.append([int(change_rows[i][0]),new, BayesDecisionRule.Tostring(tuple_)])
            
    elif(b_new < b_old):
        change_rows = DATA_train.loc[(b_new<=DATA_train[dimension]) & (DATA_train[dimension]<b_old)]
        change_rows = change_rows.reset_index()
        #change_rows.columns = ['index', 0,1,2,3,4,5,'address','bin']
        change_rows = np.array(change_rows)
        for i in range(len(change_rows)):    
            former = int(change_rows[i][7])
            label = int(change_rows[i][6])
            new_Memory[former][0] -= 1
            new_Memory[former][label+1] -= 1
            tuple_ = [int(char) for char in change_rows[i][8]]
            tuple_[dimension] += 1
            if(tuple_[dimension]==6):
                print ("error", tuple_)
                print (b_new, b_old, dimension)  
            new = BayesDecisionRule.Tuple_to_Address(tuple_, measurements, num_levels)
            
            new_Memory[new][0] += 1
            new_Memory[new][label+1] += 1
            
            change_memory_cells.append(former)
            change_memory_cells.append(new)
            change_address.append([int(change_rows[i][0]),new, BayesDecisionRule.Tostring(tuple_)])
    #print(new_Memory, np.unique(change_memory_cells), change_address)
    return new_Memory, np.unique(change_memory_cells), change_address

def Valid(b_new,bin_, new_bins, dimension):  
    if(round(b_new,4) <= new_bins[dimension][bin_-1]):
        return False
    elif(round(b_new,4) >= new_bins[dimension][bin_+1]):
        return False
    else:
        return True
    
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

def Random_Optimize_Quantizers(part1,part2, DATA_train, DATA_dev, Memory, smoothingoffset, accuracy, expectedgain, bins, measurements,seed, num_levels, data_size, Classcounts, p0,p1, Classes, actual, economic_gain, portion):
    t = 0
    nochange = 0
    Flag = False
    random.seed(seed)
    f = open('optimization.txt', 'a')
    while(expectedgain<2000000):
        t+=1
        Flag = False
        if(nochange>100):
            f.write ("No Changes Anymore!!\n")
            break
        print( "___________________round___________________________________",t)
        f.write(str(portion))
        f.write("___________________round___________________________________")
        f.write(str(t))
        f.write("\n")
        # choose a boundry to pirtrub
        dimension = random.randint(0,measurements-1)
        bin_ = random.randint(1,len(bins[dimension])-2) 
        selected_bin = bins[dimension][bin_]
        
        maxlength = max((bins[dimension][bin_]-bins[dimension][bin_-1]),(bins[dimension][bin_+1]-bins[dimension][bin_]))
        minlength = min((bins[dimension][bin_]-bins[dimension][bin_-1]),(bins[dimension][bin_+1]-bins[dimension][bin_]))
        
        #delta = random.randint(5,10)/float(1000)
        delta = 0.008
        #dom = random.randint(1,3)
        #delta = delta/dom
        #delta = (random.randint(1,10)/float(50))*(minlength)
        #delta = float(minlength)/5
        delta = round(delta,3)
        #print selected_bin, delta
        if(delta==0):
            continue
        M = int(round((maxlength)/(delta)))
        #M = 20
        b_old = bins[dimension][bin_]
        b_new = round(bins[dimension][bin_] - (delta*(M+1)),2)
        #print ("/****DELTA****", delta,"***dimension***",dimension,"***b_old,b_new***", b_old, b_new,)
        for m in range(2*M):
            new_bins = copy.deepcopy(bins)
            b_new = round(b_new + delta,3)
            if(b_new!=b_old and Valid(b_new,bin_, new_bins, dimension)):
                new_bins[dimension][bin_] = round(b_new,4)
            else:
                continue
            f.write("       **DELTA*")
            f.write(str(delta))
            f.write("\n")
            print ("/      **DELTA*", delta)#,"*dimension*",dimension,"*b_old,b_new*", b_old, b_new)
            #print ("    ",dimension, b_old, b_new )      
            #update memory cells
            new_Memory = np.array(Memory, copy=True) 
            new_Memory, Change, change_address = Update_Memory(dimension, b_new ,b_old, new_Memory, DATA_train,measurements, num_levels)
            new_Memory = Update_Decision_Rule(Change, new_Memory, smoothingoffset, data_size, Classcounts, p0,p1)
            
            prediction = BayesDecisionRule.Apply_Rule(part2, new_Memory, new_bins ,measurements, num_levels)
            new_confusion_matrix = BayesDecisionRule.Confusion_Matrix(Classes,prediction, actual)
            new_expected_gain = BayesDecisionRule.Expected_Gain(Classes, economic_gain, new_confusion_matrix,len(DATA_dev))
            new_accuracy = float(new_confusion_matrix[0][0]+new_confusion_matrix[1][1])/(new_confusion_matrix[0][0]+new_confusion_matrix[1][1]+new_confusion_matrix[0][1]+new_confusion_matrix[1][0])
            new_accuracy = round(new_accuracy,4)
            new_expected_gain = round(new_expected_gain,4)
            print (t, "new Acc",new_expected_gain," ",new_accuracy,", maxS ",expectedgain," ",accuracy)
            f.write(str(t))
            f.write(" new Acc")
            f.write(str(new_expected_gain))
            f.write(" ")
            f.write(str(new_accuracy))
            f.write(", maxS ")
            f.write(str(expectedgain))
            f.write(" ")
            f.write(str(accuracy))
            f.write("\n")
            #print new_expected_gain, expectedgain
            if(new_expected_gain > expectedgain):
                #print "HAAAAAAAAA"
                bins = new_bins
                expectedgain = new_expected_gain
                accuracy = new_accuracy
                Memory = new_Memory
                print( "YEaaaah++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print( bins)
                print( "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                f.write( "YEaaaah++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                f.write( str(bins))
                f.write("\n")
                f.write( "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                
                for i in change_address:
                    DATA_train.at[i[0],'address'] = i[1]
                    DATA_train.at[i[0],'bin'] = i[2]
                b_old = b_new
                Flag = True
            else:
                new_bins = bins
        if(Flag):
            nochange = 0
        else:
            nochange +=1
    f.close()
    return expectedgain, accuracy, bins, Memory, DATA_train

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
def InOrder_Optimize_Quantizers(part1, part2,DATA_train, DATA_dev, Memory, smoothingoffset, accuracy, expectedgain, bins, num_levels, data_size, Classcounts, p0,p1, Classes,actual, economic_gain, measurements, portion):
    t = 0
    stady = 0
    change = False
    f = open('optimization.txt', 'a')
    for dimension in range(measurements):
        THIS = len(bins[dimension])-1
        for bin_ in range(1,THIS):
            f.write(str(portion))
            f.write(" ___________________round___________________________________")
            f.write(str(t))
            f.write("\n")
            print( "___________________round____________________",t)
            t+=1
            
            selected_bin = bins[dimension][bin_]
            
            maxlength = max((bins[dimension][bin_]-bins[dimension][bin_-1]),(bins[dimension][bin_+1]-bins[dimension][bin_]))
            minlength = min((bins[dimension][bin_]-bins[dimension][bin_-1]),(bins[dimension][bin_+1]-bins[dimension][bin_]))
        
            #delta = random.randint(1,200)/1000
            #delta = (random.randint(1,10)/float(50))*(minlength)
            delta = 0.005
            delta = round(delta,3)
            if(delta==0):
                continue
            M = int(round((maxlength)/(delta)))
            #delta = random.random()/80
            #delta = random.randint(1,200)/1000
            #M = 20
            b_old = bins[dimension][bin_]
            b_new = bins[dimension][bin_-1]
            f.write("       **DELTA*")
            f.write(str(delta))
            f.write("\n")
            print ("/****DELTA****", delta)
            #print( dimension, b_old, b_new,)
            for m in range(2*M):
                new_bins = copy.deepcopy(bins)
                b_new = b_new + delta
                if(b_new!=b_old and Valid(b_new,bin_, new_bins, dimension)):
                    b_new = float(Decimal(b_new).quantize(Decimal('.00001'), rounding=ROUND_DOWN))
                    new_bins[dimension][bin_] = b_new
                else:
                    #print ("not valid")
                    continue
                print ("   ",dimension, b_old, b_new )      
                #update memory cells
                new_Memory = np.array(Memory, copy=True) 
                new_Memory, Change, change_address = Update_Memory(dimension, b_new ,b_old, new_Memory, DATA_train,measurements, num_levels)
                new_Memory = Update_Decision_Rule(Change, new_Memory, smoothingoffset, data_size, Classcounts,p0,p1)
            
                prediction = BayesDecisionRule.Apply_Rule(part2, new_Memory, new_bins, measurements, num_levels)
                new_confusion_matrix = BayesDecisionRule.Confusion_Matrix(Classes,prediction, actual)
                new_expected_gain = BayesDecisionRule.Expected_Gain(Classes, economic_gain, new_confusion_matrix,len(DATA_dev))
                new_accuracy = float(new_confusion_matrix[0][0]+new_confusion_matrix[1][1])/(new_confusion_matrix[0][0]+new_confusion_matrix[1][1]+new_confusion_matrix[0][1]+new_confusion_matrix[1][0])
                new_accuracy = round(new_accuracy,4)
                new_expected_gain = round(new_expected_gain,4)
                print ("::",t, "new Acc",new_expected_gain," ",new_accuracy,", maxS ",expectedgain," ",accuracy)
                f.write("::")
                f.write(str(t))
                f.write("new ")
                f.write(str(new_expected_gain))
                f.write(" ")
                f.write(str(new_accuracy))
                f.write(", maxz")
                f.write(str(expectedgain))
                f.write(" ")
                f.write(str(accuracy))
                f.write("\n")

                if(new_expected_gain > expectedgain):
                    bins = new_bins
                    expectedgain = new_expected_gain
                    accuracy = new_accuracy
                    Memory = new_Memory
                    #print( "YEaaaah++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    #print( bins)
                    #print( "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    f.write( "YEaaaah++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                    f.write( str(bins))
                    f.write("\n")
                    f.write( "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                
                    for i in change_address:
                        DATA_train.at[i[0], 'address'] = i[1]
                        DATA_train.at[i[0],'bin'] = i[2]
                    b_old = b_new
                    change = True
                    #return True, expectedgain, accuracy, bins, Memory, DATA_train
                else:
                 new_bins = bins
    f.close()
    if(change):
        return True, expectedgain, accuracy, bins, Memory, DATA_train
    else:
        return False,expectedgain, accuracy, bins, Memory, DATA_train