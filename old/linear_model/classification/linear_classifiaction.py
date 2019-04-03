# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:18:13 2019

@author: lee
"""
from sklearn.datasets import load_iris as lbc
import numpy as np
import numpy.linalg as lg
class Fisher(object):
    def fit(self,x,y):
        X_0=x[y.reshape(-1)==0]
        X_1=x[y.reshape(-1)==1]
        m0=X_0.sum(axis=0)/X_0.shape[0]
        m1=X_1.sum(axis=0)/X_1.shape[0]
        self.theta=lg.inv((X_0-m0).T.dot(X_0-m0)+(X_1-m1).T.dot(X_1-m1)).dot((m1-m0).T)
        

class Perceptron:
    def fit(self,x,y,max_iter=1000,initial=None):
        if initial==None:
            initial=np.ones(x.shape[1]).reshape(-1,1)
        theta=initial
        while max_iter>0:
            #寻找误分类点
            temp=x.dot(theta)*y
            x_=x[temp.reshape(-1)<0]
            y_=y[temp.reshape(-1)<0]
            print(max_iter,x_.shape[0])
            if x_.shape[0]==0:
                break
            i=np.random.randint(0,x_.shape[0])
            #更新theta值
            theta=theta+x_[i].reshape(-1,1)*y_[i]
            max_iter=max_iter-1
        self.theta=theta
        
class Generative():
    def fit(self,x,y):
        N,m=x.shape
        N_0=(y==0).sum()
        N_1=N-N_0
        mean_0=x[y.reshape(-1)==0,:].sum(axis=0)/N_0
        mean_1=x[y.reshape(-1)==1,:].sum(axis=0)/N_1
        s_0=((x[y.reshape(-1)==0]-mean_0).T).dot(x[y.reshape(-1)==0]-mean_0)
        s_1=((x[y.reshape(-1)==1]-mean_1).T).dot(x[y.reshape(-1)==1]-mean_1)
        s=(s_0+s_1)/N
        p_0=(N_0)/N
        p_1=1-p_0
        self.theta=lg.inv(s).dot((mean_0-mean_1).T)
        self.cof_=-0.5*mean_0.dot(lg.inv(s)).dot(mean_0.T)+0.5*mean_1.dot(lg.inv(s)).dot(mean_1)+np.log(p_0/p_1)
        
        
        
        
        
        
        
        
        
if __name__=="__main__":
    data_lbc=lbc()
    bol=(data_lbc.target.reshape(-1,1)!=2).reshape(-1)
    X=data_lbc.data[bol]
    y=data_lbc.target.reshape(-1,1)[bol]
#    y[y==0]=-1
    gnrt=Generative()
    gnrt.fit(X,y)