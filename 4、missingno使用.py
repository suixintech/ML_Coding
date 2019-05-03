
'''无论是打比赛还是在实际工程项目中，都会遇到数据缺失的情况，
如果数据集较小，还能在excel或者其他可视化软件大致看一下导致数据缺失的原因，
那么数据集较大时，想要探索其中规律，无疑难度也是越来越大，
今天推荐一个缺失值可视化包missingno，这个包功能很简单，只有几个方法，使用起来也特别方便，
不过它只能和pandas联合起来使用，可以说是pandas衍生功能的一大神器；'''

'''本文目录
missingno
1、安装
2、生成测试数据集
3、matrix
4、heatmap
5、bar
6、dendrogram
7、nullity_filter'''


'''1、missingno安装及使用
已经配置好环境的情况下直接使用pip安装
    pip install missingno
或者进入https://pypi.org/project/missingno/#files下载安装 
使用和Python其他库一样，直接引用'''

import missingno as msno

'''
2、生成本文测试数据集
这里使用sklearn中make_classification生成数据集，在其中增加了随机Na值，以便本文测试，数据生成代码如下所示：'''
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import missingno as msno
import matplotlib.pyplot as plt
from itertools import product
warnings.filterwarnings('ignore')
#自定义数据集，并随机产生2000个Na值，分布在各个特征之中
def getData():
    X1, y1 = make_classification(n_samples=1000, n_features=10, 

    n_classes=2,n_clusters_per_class=1, random_state=0)

    for i, j in product(range(X1.shape[0]), range(X1.shape[1])):
        if np.random.random() >= 0.8:
            xloc = np.random.randint(0, 10)
            X1[i, xloc] = np.nan
    return X1, y1

x,y=getData()

#存入pandas中
df=pd.DataFrame(x,columns=['x%s'%str(i) for i in range(x.shape[1])])
df['label']=y


'''3、matrix
两行代码，即可出现各个特征缺失值的分布情况，这里由于是生成的随机的Na值'''
msno.matrix(df)
plt.show()

'''4、heatmap
热力图，个人认为这个热力图是非常有用处的，为什么？
在正常的matplotlib中使用热力图看特征之间的相关性，
这里也同样的，使用heatmap看缺失特征之间的相关性
代码同样非常简单，如下所示'''
msno.heatmap(df)
plt.show()

'''由于这里是随机生成的Na值，因此特征之间没有强相关性，如果是有规律的缺失值，
可以看出哪些特征之间互相影响，相互缺失，这个感兴趣的同学，可以测试一下真实数据；'''


'''5、bar
条形图和matplotlib功能上没有太大的区别，
msno使用的是占比情况，df使用的count形式；'''

msno.bar(df,labels=True)
plt.show()

df.isnull().sum().plot(kind='bar')
plt.show()

'''6、dendrogram
树状图'''
msno.dendrogram(df)
plt.show()
'''树状图
a、特征越单调，越相关，距离越接近于0，从图上看，特征之间并没有相关
b、从层次聚类角度来看，这里缺失值特征基本上属于一个类别'''

'''7、nullity_filter 
从filter可以看出这个方法用来做筛选的，选择缺失值小于10%的top前2特征'''
df1=msno.nullity_filter(df,filter='top',p=0.9,n=2)
print(df.shape)
print(df1.shape)

'''missingno的方法介绍就到这里
由于水平有限，请参照指正'''



