from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# Initialize data

dataset = pd.read_csv('D:/Python/PyCharm Projects/Adatelemzések2020nyár/KAG_energydata_complete.csv',
                      usecols=range(1, 27))
'''
columns = ["Appliances", "RH_1", "T2", "RH_2", "T3", "T4", "RH_5", "T6", "RH_6", "T8", "RH_8", "T9",
           "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
'''

columns = ["Appliances", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]

df = pd.DataFrame(dataset, columns=columns)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize CatBoostRegressor
model = CatBoostRegressor(objective='RMSE', learning_rate=0.1, iterations=3000, depth=12)

# Fit model
model.fit(X_train, y_train)
# Get predictions
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 értéke CatBoostReg (train):', metrics.r2_score(y_train, y_train_pred))
print('R2 értéke CatBoostReg (test):', metrics.r2_score(y_test, y_pred))


# plt.figure(figsize=(9, 6), dpi=100)
plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.xlabel("Test adatokra kapott eredmények")
plt.ylabel("y_pred és y_test eredményei")
plt.legend()
plt.tight_layout()
plt.axis([0, len(y_test), 0, int(np.amax(y_test) + 130)])
plt.yticks(range(0, int(np.amax(y_test) + 130), 50))
plt.xticks(range(0, len(y_test), 250))
plt.grid(zorder=0, linestyle="--", alpha=0.3)
# plt.text(len(y_test) * 0.8, int(np.amax(y_test) + 20), "R2 értéke CatBoostReg: " +
 # str(round((metrics.r2_score(y_test, y_pred)), 4)))
plt.show()


