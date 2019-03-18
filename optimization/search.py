# -*- coding: utf-8 -*-
import sympy

class singleSearch:
    @staticmethod
    def golden_selection(func,l,r,interval):
        """
        区间缩小法中的黄金分割点发进行最小值点逼近
        
        Parameters
        ----------
        func:要求解的函数式
        l:区间左侧值
        r:区间右侧值
        interval:迭代停止准则
        
        Returns
        --------
        返回最小值点处的x值
        """
        #区间小于interval迭代停止
        if r-l<interval:
            return (l+r)/2
        #黄金分割比例
        rat=(5**0.5-1)/2
        #计算新的迭代点
        ll=l+(1-rat)*(r-l)
        rr=l+rat*(r-l)
        #计算新迭代点的函数值
        ll_value=func(ll)
        rr_value=func(rr)
        #选择合适的点进行下一次迭代
        if ll_value > rr_value:
            return singleSearch.golden_selection(func,ll,r,interval)
        elif ll_value < rr_value:
            return singleSearch.golden_selection(func,l,rr,interval)
        else :
            return singleSearch.golden_selection(func,ll,rr,interval)
    @staticmethod    
    def newton_method(func,start,interval):
        """
        牛顿法计算函数最小值点
        Parameters
        -------------
        func:求解最小值点的函数
        start:初始迭代点
        interval:停止准则
        
        Returns
        ----------
        返回最小值点处的x值
        """
        
        def helper(oneD,twoD,start,interval):
            t = sympy.Symbol('x')
            #计算一维导数和二维导数值
            oneD_value=oneD.subs(t,start)
            twoD_value=twoD.subs(t,start)
            #新的迭代点值
            iter_value=start-oneD_value/twoD_value
            #判断是否满足停止准则
            if abs(iter_value-start)<interval:
                return iter_value
            else:
                return helper(oneD,twoD,iter_value,interval)
        #计算函数的一阶和二阶导数
        x = sympy.Symbol('x')
        oneD=sympy.diff(func,x,1)
        twoD=sympy.diff(func,x,2)
        #helper函数求解新的迭代点
        out=helper(oneD,twoD,start,interval)
        return out
            
if __name__=="__main__":
    def cal(x):
        return 2*x**2+13*x
    a=singleSearch.golden_selection(cal,-100,50,0.01)
    x = sympy.Symbol('x')
    f=2*x**2+13*x
    b=float(singleSearch.newton_method(f,100,0.0001))