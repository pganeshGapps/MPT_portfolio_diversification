# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:24:45 2017
@author: ganesh

Purpose: "Financial Engineering : Assignment-01 "  

#Data Preprocessing:
    1)Downloaded proper data from Yahoo Finance
    2)lets say for company=Facebook or FB
            awk -F, '{print $5}' FB.csv >close_fb.csv  # extracts relevant column to a new file
            awk -F, '{print $5}' GS.csv >close_gs.csv
            awk -F, '{print $5}' AMZN.csv >close_amz.csv
            awk -F, '{print $5}' HDFCBANK.NS.csv >close_hdfcbank.csv
            awk -F, '{print $5}' GOOG.csv >close_goog.csv
            rm -f FB.csv,GS.csv,AMZN.csv,HDFC.NS.csv,GOOG.csv

"""

import numpy as np
from functools import reduce
import glob

asset=[]#(name,exp_return,variance,standard_deviation)
print("(name,exp_return,variance,standard_deviation)")
for file in glob.glob("data/*"):
    name=file.split('/')[1].split('.')[0].split('_')[1].upper()
    #print(name)
    df=np.genfromtxt(file, delimiter=',')  #dimensions=1*N
    
    df=df[1:]
    N=df.shape[0]
    returns=[0]*(N-1)  #dimension=1*(N-1)
    
    prev_val=df[0]
    for i,cur_val in enumerate(df):
        if i>0:
            returns[i-1]=(((cur_val-prev_val)/prev_val))
            prev_val=cur_val
    #print("Returns Calculated: ",len(returns))
    
    exp_return=reduce(lambda x,y: x+y,returns)/(N-1)#sum(l)/(N-1) 
    #print("Expection of Returns: ",exp_return)
    
    '''
    Variance for our Samples
    NOTE: These are samples only, 'not the Population'
    sum([(r-mean_r)**2 for r in returns])/(number_ofobserved_returns-1)
    '''
    variance=sum([(r-exp_return)**2 for r in returns])/(N-2) 
    #print("Variance: ",variance,end='\n\n')
    asset.append((name,exp_return,variance,variance**(0.5)))

print(*asset,sep='\n')

Y=[]#exp
X=[]#var
'''
Portfolio Diversification
'''
#Plotting Purpose
w=[0]*5
for w[0] in np.arange(0.0, 1.1, 0.1):
    for w[1] in np.arange(0.0, 1.1, 0.1):
        for w[2] in np.arange(0.0, 1.1, 0.1):
            for w[3] in np.arange(0.0, 1.1, 0.1):
                for w[4] in np.arange(0.0,1.1,0.1):
                    if(sum(w))==1:
                        R=sum([w[i]*asset[i][1] for i in range(5)])
                        V=sum([(w[i]**2)*(asset[i][2]) for i in range(5)])
                        X.append(V)
                        Y.append(R)

X_sd=[x**0.5 for x in X]#sd      

import matplotlib.pyplot as plt
#plt.plot(X_sd,Y)
plt.plot(X,Y)

#Optimum Weight Vector Calculation
C=np.matrix(np.diag([asset[i][2] for i in range(5)]))
O=np.matrix(np.array([1,1,1,1,1]))

C_inverse = C.I
W_transpose_min=(C_inverse*O.T)/(O*C.I*O.T)

print("\nRequired Diversification or Weight Vector:")
print(W_transpose_min)





