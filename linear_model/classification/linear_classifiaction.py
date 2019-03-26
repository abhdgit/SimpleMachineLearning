# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:18:13 2019

@author: lee
"""

import numpy as np
import numpy.linalg as lg
class Fisher(object):
    def fit(self,x,y):
        X_0=x[y.reshape(-1)==0]
        X_1=x[y.reshape(-1)==1]
        m0=X_0.sum(axis=0)/X_0.shape[0]
        m1=X_1.sum(axis=0)/X_1.shape[0]
        self.theta=lg.inv((X_0-m0).T.dot(X_0-m0)+(X_1-m1).T.dot(X_1-m1)).dot((m1-m0).T)
        
if __name__=="__main__":
    x=np.random.randint(1,100,size=1000).reshape(-1,5)
    y=x.dot(np.array([1,1,1,1,1]))+10+np.random.normal(0,9,int(1000/5))
    y=y.reshape(-1,1)
    y_=np.ones_like(y)
    y_[y>250]=0
    
    fs=Fisher()
    fs.fit(x,y_)