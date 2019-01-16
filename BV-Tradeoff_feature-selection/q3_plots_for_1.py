import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('hw1_input.csv')
print df

corr = df.corr(method='pearson')
print corr

# part 1
k = 1
arr = [df.keys()[i] for i in [0,3,4,7,9]]
for i in arr:
    for j in arr:
        if i == 'heart disease' or j == 'heart disease':
            continue
        feat1 = df[i]
        feat2 = df[j]
        plt.subplot(5,5,k)
        if i == j:
            plt.hist(feat1)
        else:
            plt.scatter(feat1, feat2)
        if k-1 in [0,5,10,15,20]:
            plt.ylabel(str(i), size=14)
        if k-1 in [20,21,22,23,24]:
            plt.xlabel(str(j), size=14)
        k += 1
plt.show()

arr = [df.keys()[i] for i in [1,2,5,6,8,10,11,12]]
k = 1
for i in arr:
    plt.subplot(2,4,k)
    plt.hist(df[i])
    plt.title(str(i))
    k += 1
plt.show()

data = np.array(df)
#data = df
data[data=='Female'] = 0
data[data=='Male'] = 1
data[data=='Abnormal'] = 0
data[data=='Angina'] = 1
data[data=='Asymptomatic'] = 2
data[data=='None'] = 3
data[data=='No'] = 0
data[data=='Yes'] = 1
data[data==' hyper'] = 0
data[data=='abnorm'] = 1
data[data=='norm'] = 2
data[data=='Down'] = 0
data[data=='Flat'] = 1
data[data=='Up'] = 2
data[data=='Fixed Defect'] = 0
data[data=='Normal'] = 1
data[data=='reversible Defect'] = 2
data[data=='Abnormal'] = 0
data[data=='Abnormal'] = 0
#data = np.array(data, dtype='float')


#print data
#print data.shape
data = np.transpose(data)
#print data
#print data.shape
#print type(data[3,0])
corr_mat = np.corrcoef(data.astype(float))

df_corr = pd.DataFrame(corr_mat)
filepath = 'corr.xlsx'
df_corr.to_excel(filepath, index=False)
print df.keys()
