# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:32:48 2019

@author: lee
"""

import numpy as np
import numpy.linalg as lg
class LeastSquares(object):
    """用于拟合线性模型的最小二乘法
    """
    def fit(self,x,y,intercept=True,l=1):
        """拟合函数
        
        Parameters
        ----------
            x:
            y:
            intercept:是否包含截距
            l:惩罚项系数(L2)
                
        Returns:
        -----------
        类LeastSquares对象
        """
        if intercept==True:
            n_,m_=x.shape
            x=np.hstack([np.ones((n_,1)),x])
            c_=lg.inv(x.T.dot(x)+l*np.identity(1+m_)).dot(x.T).dot(y)
            self.coef_ =c_[1:]
            self.intercept_=c_[0]
        
        
        
        
        
if __name__=="__main__":
    x=np.random.randint(1,100,size=1000).reshape(-1,5)
    y=x.dot(np.array([1,2,3,4,5]))+10+np.random.normal(0,9,int(1000/5))
    y=np.column_stack([y,2*y])
    lstsr=LeastSquares()
    lstsr.fit(x,y)