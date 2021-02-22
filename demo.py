import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# conf
T_stage = st.sidebar.slider("T_stage", 0, 4, 1)
st.sidebar.write('xxxxxxxxxxxxxxxx')
N_stage = st.sidebar.slider("N_stage", 0, 4, 1)
age = st.sidebar.slider("age", 1, 2, 1)
race = st.sidebar.slider("race", 1, 3, 1)
sex = st.sidebar.selectbox("sex",('female','male'),index=0)
histology = st.sidebar.slider("histology", 1, 4, 1)
grade = st.sidebar.slider("grade", 1, 4, 1)
options = st.sidebar.selectbox(
    'What are your favorite colors',
   ('Green', 'Yellow', 'Red', 'Blue'),index=0)
# str_to_int

map = {'Green':1, 'Yellow':2, 'Red':3, 'Blue':4,'male':1,'female':2}
options = map[options]
sex = map[sex]
print(options)
# 数据读取，特征标注
'''thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['bone_met'] = thyroid_train['bone_met'].apply(lambda x : +1 if x==1 else 0)
thyroid_test = pd.read_csv('test.csv', low_memory=False)
thyroid_test['bone_met'] = thyroid_test['bone_met'].apply(lambda x : +1 if x==1 else 0)
features = ['T_stage','N_stage','age','race','sex','histology','grade']
target = 'bone_met'''
#mode

#train and predict
'''RF = sklearn.ensemble.RandomForestClassifier(n_estimators=7,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
RF.fit(thyroid_train[features],thyroid_train[target])'''
with open('RF.pickle','rb') as f:
    RF = pickle.load(f)
sp = 0.0078
#figure
is_t = (1-RF.predict_proba([[T_stage,N_stage,age,race,sex,histology,grade]])[0][0]) > sp
prob = (1-RF.predict_proba([[T_stage,N_stage,age,race,sex,histology,grade]])[0][0])*1000//1/1000
st.write('is_t:',is_t,'prob is ',prob)
st.markdown('### is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))


'''im = plt.figure()
tpr,fpr,thresholds = roc_curve(thyroid_test[target],RF.predict_proba(thyroid_test[features])[:,0])
ax1=im.add_subplot(1,1,1)
ax1.plot(fpr, tpr,label = 'RandomForest')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(np.arange(0,1.05,0.05),np.arange(0,1.05,0.05),'--')
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
print(auc(fpr,tpr))
plt.legend()
st.pyplot(im)'''

