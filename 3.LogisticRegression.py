#!/usr/bin/python3
#-*-coding:utf-8 -*-
# __Author__  : 随心
# __File__    : LogisticRegression.py
# __Software__: PyCharm


'''
    逻辑回归的实现及注意事项:
    1、要求单个label服从伯努利分布，整体样本服从二项分布,属于广义线性模型中的一种
    2、使用多种优化方法实现逻辑回归(二分类 and 多分类)，使用以下数据集
    def getData():
        from sklearn.datasets import make_classification
        X1, y1 = make_classification(n_samples=200,n_features=10,n_classes=2,n_clusters_per_class=1,random_state=0)
        X2, y2 = make_classification(n_samples=200, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=0)
        return X1,y1,X2,y2
'''

class LogitGD:
    def __init__(self):
        self.X1, self.y1, _, _ = getData()
        self.y1= self.y1.reshape(200, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_split_test(self.X1, self.y1)

    """激活函数"""
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    """代价函数"""
    def cost(self,theta,X,y):
        return -np.mean(y*np.log(self.sigmoid(X@theta))+(1-y)*np.log(1-self.sigmoid(X@theta)))

    """梯度"""
    def gradient(self,theta, X, y):
        parameters=int(theta.shape[0])
        grad=np.zeros(parameters)
        error=self.sigmoid(X @ theta) - y
        for i in range(parameters):
            term=np.mean(np.multiply(error,X[:,i].reshape(140,1)))
            grad[i]=term
        return grad

    """批量梯度下降"""
    def gradient_decent(self,X, y, max_iter=200, learning_rate=0.005):
        theta=np.zeros(X.shape[1]).reshape(X.shape[1], 1)
        costs=[self.cost(theta,X,y)]
        _theta=theta.copy()
        err = 0.00000001
        for _ in range(max_iter):
            _theta=_theta-learning_rate*self.gradient(_theta,X,y).reshape(10,1)
            costs.append(self.cost(_theta,X,y))
            if np.abs(costs[-1]-costs[-2])<err:
                break
        return _theta,costs

    def train_split_test(self,X,y,ratio=0.7):
        length=int(len(X)*ratio)
        X_train=X[0:length,:]
        y_train=y[0:length]
        X_test=X[length:,:]
        y_test=y[length:]
        return X_train,X_test,y_train,y_test

    def fit(self):
        self.final_theta,costs = self.gradient_decent(self.X_train, self.y_train)


    def predict(self,X_test):
        y_predict=X_test@self.final_theta
        y_predict=self.sigmoid(y_predict)
        return y_predict

    def score(self):
        right_count=0
        y_predict=self.predict(self.X_test)
        for x,y in zip(self.y_test,y_predict):
            if (x==1 and y>=0.5) or (x==0 and y<0.5):
                right_count+=1
        return right_count/len(self.X_test)


class LogitSGD:
    def __init__(self):
        self.X1, self.y1, _, _ = getData()
        self.y1= self.y1.reshape(200, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_split_test(self.X1, self.y1)

    """激活函数"""
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    """代价函数"""
    def cost(self,theta,X,y):
        return -np.mean(y*np.log(self.sigmoid(X@theta))+(1-y)*np.log(1-self.sigmoid(X@theta)))

    """随机梯度下降"""

    def s_gradient(self,theta, X, y):
        parameters = int(theta.shape[0])
        grad = np.zeros(parameters)
        error = self.sigmoid(X @ theta) - y
        for i in range(parameters):
            term = error * X[i]
            grad[i] = term
        return grad

    def s_gradient_decent(self,X, y, max_iter=200, learning_rate=0.005):
        theta = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
        costs = [self.cost(theta, X, y)]
        m = len(X)
        _theta = theta.copy()
        err = 0.00000001
        for _ in range(max_iter):
            for j in range(m):
                X_1 = X[j, :]
                y_1 = y[j]
                _theta = _theta - learning_rate * self.s_gradient(_theta, X_1, y_1).reshape(10, 1)
                costs.append(self.cost(_theta, X, y))
                if np.abs(costs[-1] - costs[-2]) < err:
                    break
        return _theta, costs

    def train_split_test(self,X, y,ratio=0.7):
        length = int(len(X) *ratio)
        X_train = X[0:length, :]
        y_train = y[0:length]
        X_test = X[length:, :]
        y_test = y[length:]
        return X_train, X_test, y_train, y_test

    def fit(self):
        self.final_theta,costs = self.s_gradient_decent(self.X_train, self.y_train)

    def predict(self,X_test):
        y_predict = X_test @ self.final_theta
        y_predict = self.sigmoid(y_predict)
        return y_predict

    def score(self):
        right_count = 0
        y_predict = self.predict(self.X_test)
        for x, y in zip(self.y_test, y_predict):
            if (x == 1 and y >= 0.5) or (x == 0 and y < 0.5):
                right_count += 1
        return right_count / len(self.X_test)



class LogitPso:
    def __init__(self):
        self.X1, self.y1, _, _ = getData()
        self.y1= self.y1.reshape(200, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_split_test(self.X1, self.y1)
        self.w=0.8
        self.c1,self.c2=2,2
        self.r1,self.r2=0.6,0.3
        self.pN=30
        self.dim=11
        self.max_iter=3000
        self.X=np.zeros((self.pN,self.dim))
        self.V=np.zeros((self.pN,self.dim))
        self.pbest=np.zeros((self.pN,self.dim))
        self.gbest=np.zeros((1,self.dim))
        self.p_fit=np.zeros(self.pN)
        self.fit=1e+6

    def train_split_test(self,X, y,ratio=0.7):
        length = int(len(X) *ratio)
        X_train = X[0:length, :]
        y_train = y[0:length]
        X_test = X[length:, :]
        y_test = y[length:]
        return X_train, X_test, y_train, y_test

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def costFunction(self,theta,x,y):
        return -np.mean(y*np.log(self.sigmoid(np.insert(x,0,1,axis=1)@theta))+(1-y)*np.log(1-self.sigmoid(np.insert(x,0,1,axis=1)@theta)))

    def initPopulation(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j]=np.random.uniform(0,1)
                self.V[i][j]=np.random.uniform(0,1)
            self.pbest[i]=self.X[i]
            cost=self.costFunction(self.X[i],self.X_train,self.y_train)
            self.p_fit[i]=cost
            if (cost<self.fit):
                self.fit=cost
                self.gbest=self.X[i]


    def fitModel(self):
        self.initPopulation()
        costValue=[]
        for i in range(self.max_iter):
            for j in range(self.pN):
                cost=self.costFunction(self.X[j],self.X_train,self.y_train)
                if (cost<self.p_fit[j]):
                    self.pbest[j]=self.X[j]
                    self.p_fit[j]=cost
                    if (self.p_fit[j]<self.fit):
                        self.gbest=self.X[j]
                        self.fit=self.p_fit[j]
            for k in range(self.pN):
                self.V[k]=self.w*self.V[k]+self.c1*self.r1*(self.pbest[k]-self.X[k])+self.c2*self.r2*(self.gbest-self.X[k])
                self.X[k]=self.X[k]+self.V[k]
            costValue.append(self.fit)
        return self.gbest,costValue
    
    

    

if __name__=="__main__":
    demo=LogitSGD()
    demo.fit()
    s1=demo.score()
    demo1=LogitSGD()
    demo1.fit()
    s2=demo1.score()
    print("梯度下降Score为:%s,随机梯度下降Score为:%s"%(round(s1,3),round(s2,3)))
