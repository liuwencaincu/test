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

#应用标题
st.title('A random forest model for predicting the risk of bone metastases in thyroid cancer patients based on demographic and clinical characteristics')



# conf

age = st.sidebar.selectbox("Age",('<50','>=50'),index=0)
sex = st.sidebar.selectbox("sex",('female','male'),index=1)
race = st.sidebar.selectbox("Race",('Black','Other','White'),index=0)
histology = st.sidebar.selectbox("Histology",('ATC','FTC','MTC','PTC'),index=3)
T_stage = st.sidebar.slider("T stage", 0, 4, 0)
N_stage = st.sidebar.slider("N stage", 0, 1, 0)
grade = st.sidebar.slider("Grade", 1, 4, 1)

# str_to_int

map = {'<50':1,'>=50':2,'male':1,'female':2,'Black':1,'Other':2,'White':3,'ATC':1,'FTC':2,'MTC':3,'PTC':4}

age = map[age]
sex = map[sex]
race = map[race]
histology = map[histology]

# 数据读取，特征标注
#thyroid_train = pd.read_csv('train.csv', low_memory=False)
#thyroid_train['bone_met'] = thyroid_train['bone_met'].apply(lambda x : +1 if x==1 else 0)
#thyroid_test = pd.read_csv('test.csv', low_memory=False)
#thyroid_test['bone_met'] = thyroid_test['bone_met'].apply(lambda x : +1 if x==1 else 0)
#features = ['T_stage','N_stage','age','race','sex','histology','grade']
#target = 'bone_met'
#mode

#train and predict
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=7,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
#RF.fit(thyroid_train[features],thyroid_train[target])

#读之前存储的模型
with open('RF.pickle', 'rb') as f:
    RF = pickle.load(f)


sp = 0.0078
#figure
is_t = (1-RF.predict_proba([[T_stage,N_stage,age,race,sex,histology,grade]])[0][0]) > sp
prob = (1-RF.predict_proba([[T_stage,N_stage,age,race,sex,histology,grade]])[0][0])#*1000//1/1000
#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))
if is_t == True:
    result = 'A high risk of bone metastasis !'
else:
    result = 'A low risk of bone metastasis .'
st.markdown('## Predict:  '+str(result))
st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行
st.text("")
st.text("")
st.title("")
st.title("")
st.title("")

st.info('Thank you for using! This is a purely informational message. If you click the button below, there will be celebration!')
if st.button('Click for celebration'):
    st.balloons()
    st.balloons()


