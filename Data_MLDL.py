import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn import model_selection, svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

data_df = pd.read_csv('train.csv')
target = data_df['credit'].copy()

y_label = data_df['credit']
data_df = data_df.drop(['credit'],axis=1)

#자동으로 num과 cat 변수 갈라서 df 생성
data_df_cat = data_df.select_dtypes(include=np.object)
data_df_num = data_df.select_dtypes(exclude=np.object)

# cat인데 숫자로 되어있는 변수 따로 cat df로 넣어주기
tmp = data_df[['FLAG_MOBIL','work_phone','phone','email']]
data_df_cat = pd.concat([data_df_cat, tmp], axis = 1)
data_df_cat.head(2)

#cat인데 num df에 들어간 변수 num df에서 drop해주기
data_df_num = data_df_num.drop(columns=['FLAG_MOBIL','work_phone','phone','email', 'index'], axis = 1)

""" Pipeline """
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_df,
                                                                    target,
                                                                   test_size = 0.2,
                                                                   random_state=0)

numeric_features = data_df_num.columns
numeric_transformer = StandardScaler() # cf) RobustScaler

categorical_features = data_df_cat.columns
categorical_transformer = OneHotEncoder(categories='auto', handle_unknown='ignore') # categories='auto' : just for ignoring warning messages

preprocessor = ColumnTransformer(
    transformers=[ # List of (name, transformer, column(s))
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

preprocessor_pipe = Pipeline(steps=[('preprocessor', preprocessor)]) # preprocessing-only
preprocessor_pipe.fit(x_train)

x_train_transformed = preprocessor_pipe.transform(x_train)
x_test_transformed = preprocessor_pipe.transform(x_test)

""" SVM """
lin_clf = svm.LinearSVC()
lin_clf.fit(x_train_transformed, y_train)
accuracy = lin_clf.score(x_test_transformed, y_test)
print("model score:", round(accuracy, 4))

""" GradientBoostingClassifier """
model = GradientBoostingClassifier(n_estimators=200, random_state=0)
model.fit(x_train_transformed, y_train)
accuracy = model.score(x_test_transformed, y_test)
print("model score:", round(accuracy, 4))

""" XGBClassifier """
xgb_model = XGBClassifier()
xgb_model.fit(x_train_transformed, y_train)
accuracy = xgb_model.score(x_test_transformed, y_test)
print("model score:", round(accuracy, 4))