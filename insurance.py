#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#%%
df = pd.read_csv("C:/Users/Admin/Desktop/Code GYM/TH/insurance.csv")
df.head()
#%%
df.info()
#%%
df.describe()
#%%
fig, axis = plt.subplots(1, 3, figsize=(18, 4))
sns.scatterplot(data = df, x="TV", y = "Sales", ax=axis[0])
sns.scatterplot(data = df, x="Radio", y = "Sales", ax=axis[1])
sns.scatterplot(data = df, x="Newspaper", y = "Sales", ax=axis[2])
#%%
sns.boxplot(data=df)
#%%
coll_na = df.columns[df.isnull().any()]
#%%
print("Kết luận: Không có dữ liệu khuyết thiếu nên không cần xử lý")
#%%
y = df["Sales"]
df_iv = df.drop("Sales", axis=1)
#%%
df_cor = df_iv.corr(method="pearson")

# Tạo ra mặt nạ loại bỏ tam giá cái 1/2 phần trên của heatmap
mask = np.triu(np.ones_like(df_cor, dtype=bool))

# Tạo bảng màu đẹp hơn
cmap =  sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(df_cor, vmin=-1, vmax=1, mask=mask, cmap=cmap, center=0, annot=True , square=True, linewidths=0.5, cbar_kws={"shrink": 0.5, "orientation": "vertical"})
#%%
X_train, X_test, y_train, y_test = train_test_split(df_iv, y, test_size=0.2, random_state=42)
#%%
type(X_train)
#%%
X_train.info()
#%%
model_r = LinearRegression()
model_r.fit(X_train, y_train)
#%%
scr = model_r.score(X_test, y_test)
scr
#%%
y_pred = model_r.predict(X_test)
#%%
type(y_pred)
#%%
print("y_pred", y_pred)
print("y_test", y_test.to_numpy())
#%%
type(y_test)
#%%
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)