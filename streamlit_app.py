import streamlit as st
import pandas as pd
import numpy as np
from streamlit_javascript import st_javascript
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

st.header("Isolation Forest", divider='gray')

df = pd.read_json("data.json")
st.markdown('#### 1. 데이터 살펴보기')
st.dataframe(df.style.highlight_max(axis=0), hide_index = True)
df_test = df.iloc[:,4:]

clf=IsolationForest(n_estimators=50, max_samples=50, contamination=float(0.1), 
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=None, verbose=0)
# 50개의 노드 수, 최대 50개의 샘플
# 0.1%의 outlier 색출.

st.markdown('<p style="padding-top:100px"></p>', unsafe_allow_html=True)
st.markdown('#### 2. sklearn에서 제공하는 Isolation Forest로 정상치, 이상치 구분')
st.markdown('(오염물질 구분없이 전체오염도를 기준으로 이상치 탐색)')

code = '''
from sklearn.ensemble import IsolationForest

clf=IsolationForest(n_estimators=50, max_samples=50, contamination=float(0.1), max_features=1.0, bootstrap=False, n_jobs=-1, random_state=None, verbose=0)
# 50개의 노드 수, 최대 50개의 샘플
# 0.1%의 outlier 색출.'''
st.code(code, language="python")

clf.fit(df_test)
pred = clf.predict(df_test)
df_test['anomaly']=pred
outliers=df_test.loc[df_test['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
st.write(df_test['anomaly'].value_counts())


st.markdown('<p style="padding-top:100px"></p>', unsafe_allow_html=True)
st.markdown('#### 3. 차원의 수를 줄인 후 3D로 표시')
st.markdown('( 측정 지표를 정규화하고 PCA에 적합시켜 차원의 수를 줄인 후 3D로 표시)')

pca = PCA(n_components=3) 
scaler = StandardScaler()
X = scaler.fit_transform(df_test)
X_reduce = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
           lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
st.pyplot(fig)


st.markdown('<p style="padding-top:100px"></p>', unsafe_allow_html=True)
st.markdown('#### 4. 차원의 수를 줄인 후 2D로 표시')

pca01 = PCA(2)
pca01.fit(df_test)
res=pd.DataFrame(pca01.transform(df_test))
Z = np.array(res)
# plt.contourf( Z, cmap=plt.cm.Blues_r)
fig01 = plt.figure()
b1 = plt.scatter(res[0], res[1], c='green',
                 s=20,label="normal points")
b1 =plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='green',s=20,  edgecolor="red",label="predicted outliers")
plt.legend(loc="upper right")
plt.show()
st.pyplot(fig01)