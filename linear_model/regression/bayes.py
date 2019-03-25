# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:44:12 2019

@author: lee
"""
import numpy as np
import numpy.linalg as lg
class BayesMethod(object):
    """利用贝叶斯方法拟合线性回归模型
    假定w的先验分布服从(0,alpha*I)的高斯分布
    数据t服从N(t|y(x),1/beta)的分布
    可知后验分布p(w|x,t)=N(m,s)
    """
    def fit(self,x,y):
        alpha=1
        self.alpha=alpha
        beta=1
        self.beta=beta
        n_,m_=x.shape
        x=np.hstack([np.ones((n_,1)),x])
        s=(alpha*np.identity(m_+1)+beta*x.T.dot(x))
        m=beta*lg.inv(s).dot(x.T).dot(y)
        self.mean_=m
        self.vars_=lg.inv(s)
    def predict(self,x):
        x=np.insert(x,0,1)
        mean_=((self.mean_*x).sum())
        var_=1.0/self.beta+x.T.dot(self.vars_).dot(x)
        return([mean_,var_])
if __name__=="__main__":
    #生成数据
    x=np.random.randint(1,100,size=1000).reshape(-1,5)
    y=x.dot(np.array([1,2,3,4,5]))+10+np.random.normal(0,9,int(1000/5))
    bm=BayesMethod()
    bm.fit(x,y)
    bm.predict(np.array([1,1,1,1,1]))