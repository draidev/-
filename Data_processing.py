import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 파일 읽어오기
data_df = pd.read_csv('train.csv')
target = data_df['credit'].copy()

""" 01 전처리 전 데이터 시각화 """
y_label = data_df['credit']
data_df = data_df.drop(['credit'],axis=1)

#자동으로 num과 cat 변수 갈라서 df 생성
data_df_cat = data_df.select_dtypes(include=np.object)
data_df_num = data_df.select_dtypes(exclude=np.object)

# cat인데 숫자로 되어있는 변수 따로 cat df로 넣어주기
tmp = data_df[['FLAG_MOBIL','work_phone','phone','email']]
data_df_cat = pd.concat([data_df_cat, tmp], axis = 1)

#cat인데 num df에 들어간 변수 num df에서 drop해주기
data_df_num = data_df_num.drop(columns=['FLAG_MOBIL','work_phone','phone','email', 'index'], axis = 1)

# cat과 num 변수 빠짐없이 잘 divide 되었는지 확인하기
len(data_df_cat.columns) + len(data_df_num.columns) == len(data_df.columns) - 1
data_df_v=pd.concat([data_df_cat,y_label],axis=1).astype(object)
# Plot a count of the categories from each categorical feature split by our prediction class: salary - predclass.
def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            
plot_bivariate_bar(data_df_v, hue='credit', cols=3, width=20, height=12, hspace=0.4, wspace=0.5)

""" 02 데이터 전처리 """
""" occyp_type열의 결측치 채우기 """
# income_type의 value 리스트
income_list = list(set(data_df['income_type'].values))

#income_type별로 가장 많은 occyp_type값을 리스트로 만듬
occyp_list = []
for i in income_list:
    occyp_list.append(data_df[data_df['income_type']== i]['occyp_type'].value_counts().index[0])

# 직업 종류 결측치 income_type으로 채우기
for i in data_df[data_df['occyp_type'].isnull()].index:
    if data_df['DAYS_EMPLOYED'][i] > 0:
        data_df['occyp_type'][i] = 'Unemployed'
    else:
        for income in income_list:
            if income == data_df['income_type'][i]:
                data_df['occyp_type'][i] = data_df[data_df['income_type']== income]['occyp_type'].value_counts().index[0]
print(data_df['occyp_type'])
print(data_df['occyp_type'].value_counts())