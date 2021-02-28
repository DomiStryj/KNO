import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Wczytanie pliku
colnames = ['species', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']

dataframe = pd.read_csv('penguins_cleaned.csv', sep=',', header=None, names=colnames)

dataframe = dataframe.replace({'species': {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}})
X = dataframe.iloc[:, 1:]
X = pd.get_dummies(X)
y = dataframe.iloc[:, 0].values

from sklearn.svm import SVC  # "Support vector classifier"
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state=42)
model = SVC(kernel='linear')
model.fit(train_features, train_labels)

svc_prediction = model.predict(test_features)
print('Accuracy SVM:', accuracy_score(test_labels, svc_prediction))

