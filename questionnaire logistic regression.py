# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv("questionnaire logistic regression.csv")
scaler = StandardScaler()

param_grid_lr = {
    'max_iter': [20, 50, 100, 200, 500, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'class_weight': ['balanced']
}

df.drop(["user", "std_reaction_time"], axis=1, inplace=True)

acc = df["accuracy"]
mrt = df["mean_reaction_time"]
# stat = df["status"]
rate = df["rating"]

accs = np.std(acc)
mrts = np.std(mrt)
# stats = np.std(stat)
rates = np.std(rate)

aug_df = pd.DataFrame()

for _ in range(1):
    for _, row in df.iterrows():
        temp = {
            'accuracy': np.random.uniform(accs),
            'mean_reaction_time': row['mean_reaction_time'] + np.random.uniform(mrts),
            # 'status': row['status'] + np.random.uniform(stats),
            'rating': round(row['rating'] + np.random.uniform(rates) - 1),
            }
        aug_df = aug_df.append(temp, ignore_index=True)

df = df.append(aug_df)

df["status"] = ["0" if ele < 6 else "1" for ele in df["rating"]]
df['status'] = df['status'].astype('int')

Y = df['status'].values
X = df.drop(labels = ["status"], axis=1)
X_scaled = scaler.fit_transform(X)

# x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=20)
# model = LogisticRegression()
# model.fit(x_train, y_train)
# predict_test = model.predict(x_test)

# accuracy = accuracy_score(y_test, predict_test)
# print("Accurracy = ", (accuracy * 100.0), "%")

grid_search = GridSearchCV(estimator=LogisticRegression(random_state=1234), param_grid=param_grid_lr, verbose=1, cv=10, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

predict_test = best_model.predict(x_test)
accuracy = accuracy_score(y_test, predict_test)
precision = precision_score(y_test, predict_test)
recall = recall_score(y_test, predict_test)
f1 = f1_score(y_test, predict_test)

print("Accuracy = ", (accuracy * 100.0), "%")
print("Precision = ", (precision * 100.0), "%")
print("Recall = ", (recall * 100.0), "%")
print("F1 Score = ", (f1 * 100.0), "%")
