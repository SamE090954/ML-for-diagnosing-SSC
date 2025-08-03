#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score

#Upload data into a DataFrame
file_path_expressions = #Path to data goes here
df_start = pd.read_csv(file_path_expressions, sep = "\t")
df_flip = df_start.T
df_flip.columns = df_flip.iloc[0]
df = df_flip[1:]

#Remove noisy data and separate into target and input groups
columns_ignore = ['gene', 'Gene', 'Condition']
X = df.drop(columns_ignore, axis =1)
y = df['Condition']

#Preprocessing
selector = SelectFromModel(estimator = RandomForestClassifier(n_estimators = 100, random_state = 42), max_features = 45).fit(X,y) 
feature_mask = selector.get_support()
selected_column_names = X.columns[feature_mask]
new = selector.transform(X)
X = pd.DataFrame(new, columns= selected_column_names)

#Instantiate model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#Separate data into test and training sets
#Fit and test model
#Record accuracy scores
fold = StratifiedKFold(n_splits = 4)
accuracy_list =[]
f1_list = []
for train_index, test_index in fold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    f = f1_score(y_test, y_pred, pos_label = 'SSc')
    f1_list.append(f)
    accuracy_list.append(rf_classifier.score(X_test, y_test))

#Cross validate accuracy score
score = cross_val_score(rf_classifier , X,y)
print("\nCross_val_score indicates %0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))

#Print accuracy and f1 scores
print('\nMaximum Accuracy: ',
	max(accuracy_list)*100)
print('\nMinimum Accuracy: ',
	min(accuracy_list)*100)
print('\nOverall Accuracy: ',
	statistics.mean(accuracy_list)*100)
print('\nMaximum F1 Score: ',
	max(f1_list)*100)
print('\nMinimum F1 Score: ',
	min(f1_list)*100)
print('\nOverall F1 score: ',
	statistics.mean(f1_list)*100)

#Find important features
importances = rf_classifier.feature_importances_ 
feature_names = X_train.columns
df_feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
df_feature_importance = df_feature_importance.sort_values(by='importance', ascending = False)

#Plotsfeature importance
plt.figure(figsize=(12,9))
sns.barplot(x='importance', y ='feature',data=df_feature_importance)
plt.title('Random Forest Feature Importance When Diagnosing SSc')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

#Print feature importance list
print(df_feature_importance)

#Measure AUC score
y_prob = rf_classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label = 'SSc')
auc_score = roc_auc_score(y_test,y_prob)

#Plot AUC score
plt.figure(figsize = (10,6))
plt.plot(fpr,tpr, label=f'Random Forest (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color = 'gray', label = 'Random Forest Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend()
plt.grid(True)
plt.show()

