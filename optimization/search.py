# -*- coding: utf-8 -*-


class singleSearch:
    @staticmethod
    def golden_selection(func,l,r,interval):
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
            
if __name__=="__main__":
    def cal(x):
        return 2*x**2+20*x
    a=singleSearch.golden_selection(cal,-100,50,0.01)