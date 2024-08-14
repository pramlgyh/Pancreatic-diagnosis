import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus']=False
#处理图像显示中文的问题
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
from sklearn.metrics import *

y_pred = [0.99, 0.998, 0.993, 0.897, 0.995, 0.897, 0.004, 0.003, 0.001, 0.002]
#预测值
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#实际值
fpr_Nb, tpr_Nb, _ = roc_curve(y_true, y_pred)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值
plt.figure(figsize=(10,8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_Nb, tpr_Nb,"r",linewidth = 3)
plt.grid()
plt.xlabel("假正率FPR")
plt.ylabel("真正率TPR")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("ROC曲线")
plt.text(0.15,0.9,"AUC = "+str(round(aucval,4)))
plt.show()