#!/usr/bin/python3
#-*-coding:utf-8 -*-
# __Author__  : 随心
# __File__    : LinearRegression.py
# __Software__: PyCharm

'''
    线性回归的假设条件：
    1、样本独立，即每个预测样本之间没有依赖关系；
    2、残差e要服从正态分布，即y_true-y_pred的残差需要服从高斯分布；
    3、特征之间独立，即特征之间需要独立，如果不独立(共线性)会造成系数权重之和为单特征权重且权重方差较大，同时会造成模型预测结果震荡，不稳定；
    4、样本数需要大于特征数，如果特征数量大于样本数量，通过最小二乘法无法求矩阵的逆，通过其他优化方式得到的最优结果非唯一解，造成模型偏差较大；
    5、残差e要求方差齐性，即残差不随观测变量的变化而变化；
    6、自变量与因变量之间呈线性关系；
    使用最小二乘法、梯度下降、随机梯度下降、PSO粒子群算法及牛顿法实现多元线性回归，数据集使用如下数据集
    from sklearn import datasets
    data=datasets.load_diabetes()
'''


from contextlib import contextmanager
from time import strftime
import time
import numpy as np
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')

@contextmanager
def timeSchedule(message: str):
    """ Time Schedule
    """
    print('[INFO {}][{}] Start ...'.format(strftime('%Y-%m-%d %H:%M:%S'), message))
    start_time = time.time()
    yield
    print('[INFO {}][{}] End   ...'.format(strftime('%Y-%m-%d %H:%M:%S'), message))
    print('[INFO {}][{}] Cost {:.2f} s'.format(strftime('%Y-%m-%d %H:%M:%S'), message, time.time() - start_time))


class LinearRegressionLSM:
    '''
        线性回归-最小二乘法
    '''

    def __init__(self):
        self.x,self.y=datasets.load_diabetes()['data'],datasets.load_diabetes()['target']

    def fit(self):
        self.x=np.insert(self.x,0,1,axis=1)
        self.x_=np.linalg.inv(self.x.T.dot(self.x))
        return self.x_.dot(self.x.T).dot(self.y)

    def predict(self,x):
        w=self.fit()
        w=w.reshape(-1,1)
        return np.sum(w.T*x,axis=1)



class LinearRegressionGD:
    '''
        线性回归-梯度下降
    '''
    def __init__(self):
        self.data=datasets.load_diabetes()
        self.x, self.y= datasets.load_diabetes()['data'], datasets.load_diabetes()['target']
        self.w=np.zeros((self.x.shape[1],1))
        self.b=np.array([0.0])
        self.step=200000

    def costFunction(self,theta0,theta1,x,y):
        ddd=((np.sum(theta1.T*x,axis=1)+theta0)-y)**2
        J=np.sum(ddd,axis=0)
        return J/(2*x.shape[0])

    def partTheta0(self,theta0,theta1,x,y):
        h=theta0+np.sum(theta1.T*x,axis=1)
        diff=h-y
        partial=diff.sum()/x.shape[0]
        return partial

    def partTheta1(self,theta0,theta1,x,y):
        partials=[]
        for i in range(x.shape[1]):
            h=theta0+np.sum(theta1.T*x,axis=1)
            diff=(h-y)*x[:,i]
            partial=diff.sum()/x.shape[0]
            partials.append(partial)
        return np.array(partials).reshape(x.shape[1],1)

    def fit(self,x,y,aph=0.01):
        theta0, theta1=self.b,self.w
        counter=0
        c=self.costFunction(theta0,theta1,x,y)
        costs=[c]
        c1=c+10
        err=0.00000001
        while (np.abs(c-c1)>err) and (counter<self.step):
            c1=c
            update_theta0=aph*self.partTheta0(theta0,theta1,x,y)
            update_theta1=aph*self.partTheta1(theta0,theta1,x,y)
            theta0-=update_theta0
            theta1-=update_theta1
            c=self.costFunction(theta0,theta1,x,y)
            costs.append(c)
            counter+=1
        return theta0,theta1,counter

    def predict(self,x):
        w0,w1,c=self.fit(self.x,self.y)
        return (np.sum(w1.T*x,axis=1)+w0).sum()


xx=np.array([0.03807591,0.05068012,0.06169621,0.02187235,-0.0442235,-0.03482076,-0.04340085,-0.00259226,0.01990842,-0.01764613])
# print(LinearRegressionGD().predict(xx))


class LinearRegressionSGD:
    '''
        线性回归-随机梯度下降
    '''

    def __init__(self):
        self.data = datasets.load_diabetes()
        self.x, self.y =datasets.load_diabetes()['data'], datasets.load_diabetes()['target']
        self.w = np.zeros((self.x.shape[1], 1))
        self.b = np.array([0.0])


    def costFunction(self,theta0,theta1,x,y):
        h=((theta0+np.sum(theta1.T*x,axis=1))-y)**2
        J=np.sum(h,axis=0)/2
        return J


    def partTheta0(self,theta0,theta1,x,y):
        h=theta0+np.sum(theta1.T*x,axis=1)
        diff=(h-y)
        return diff

    def partTheta1(self,theta0,theta1,x,y):
        partials=[]
        h=theta0+np.sum(theta1.T*x,axis=1)
        for i in range(x.shape[0]):
            diff=(h-y)*x[i]
            partials.append(diff)
        return np.array(partials).reshape(x.shape[0],1)

    def fit(self):
        #初始化第1个样本点的损失函数值
        c=self.costFunction(self.b,self.w,self.x[0],self.y[0])
        partTheta0=self.b
        partTheta1=self.w
        #遍历所有样本点，计算对应损失函数值
        step=0
        aph=0.01
        #参与计算的总样本数
        totalstep=10000
        for i in range(1,self.x.shape[0]):
            step+=1
            c1=c
            x,y=self.x[i,:],self.y[i]
            updateTheta0=self.partTheta0(partTheta0,partTheta1,x,y)
            updateTheta1=self.partTheta1(partTheta0,partTheta1,x,y)
            partTheta0-=aph*updateTheta0
            partTheta1-=aph*updateTheta1
            c=self.costFunction(partTheta0,partTheta1,x,y)
            if np.abs(c-c1)<=0.000000001 and step<totalstep:
                return partTheta0,partTheta1,i
            if step>=totalstep:
                return partTheta0,partTheta1,step
        return partTheta0, partTheta1, step


    def predict(self,x):
        #训练所有样本点的结果
        w0, w1,step= self.fit()
        #进行预测
        return (np.sum(w1.T * x, axis=1) + w0).sum()


# print(LinearRegressionSGD().predict(xx))


class LinearRegressionNM:
    '''
        线性回归-牛顿法
    '''
    def __init__(self):
        self.data = datasets.load_diabetes()
        self.x, self.y =datasets.load_diabetes()['data'], datasets.load_diabetes()['target']
        #插入数据1
        self.x=np.insert(self.x,0,1,axis=1)


    def computeHessianinv(self,x):
        #注意区别与最小二乘法之间的推导区别
        return np.linalg.inv(x.T.dot(x))


    def fit(self):
        return self.computeHessianinv(self.x).dot(self.x.T).dot(self.y)

    def predict(self,x):
        w=self.fit()
        return w[0]+np.sum(w[1:].T*x,axis=1)

# LinearRegressionNM().fit()
class LiearRegressionPSO:
    '''
        线性回归-PSO粒子群算法
    '''
    def __init__(self):
        self.data = datasets.load_diabetes()
        self.x, self.y = datasets.load_diabetes()['data'], datasets.load_diabetes()['target']
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.5
        self.r2 = 0.5
        self.pN = 30  #粒子数量
        self.dim = 11  #参数个数
        self.max_iter = 2000  # 最大迭代次数
        self.X = np.zeros((self.pN, self.dim))  #粒子位置
        self.V = np.zeros((self.pN, self.dim)) #粒子速度
        self.pbest = np.zeros((self.pN, self.dim)) #粒子最佳解
        self.gbest = np.zeros((1, self.dim)) #全局最优解
        self.p_fit = np.zeros(self.pN)#粒子最佳适应值
        self.fit = 100000000  #全局适应值初始值


    #损失函数，适应函数
    def costFunction(self,x):
        h=np.sum(x.T*np.insert(self.x,0,1,axis=1),axis=1)-self.y
        diff=h**2
        return diff.sum()/self.x.shape[0]

    #粒子群初始化
    def initPopulation(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = np.random.uniform(0, 1)
                self.V[i][j] = np.random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            cost = self.costFunction(self.X[i])
            self.p_fit[i] = cost
            if (cost < self.fit):
                self.fit = cost
                self.gbest = self.X[i]

    def fitModel(self):
        #初始化粒子群
        self.initPopulation()
        costVale=[]
        for i in range(self.max_iter):
            for j in range(self.pN):
                cost=self.costFunction(self.X[j])
                if (cost<self.p_fit[j]):
                    self.pbest[j]=self.X[j]
                    self.p_fit[j]=cost
                    if (self.p_fit[j]<self.fit):
                        self.gbest=self.X[j]
                        self.fit=self.p_fit[j]
            for k in range(self.pN):
                self.V[k]=self.w*self.V[k]+self.c1*self.r1*(self.pbest[k]-self.X[k])+self.c2*self.r2*(self.gbest-self.X[k])
                self.X[k]=self.X[k]+self.V[k]
            costVale.append(self.fit)
        return self.gbest,costVale

    def predict(self,x):
        w,cost=self.fitModel()
        return w[0]+np.sum(w[1:].reshape(-1,1).T*x,axis=1)

# clf=LiearRegressionPSO()
# print(clf.predict(xx))









