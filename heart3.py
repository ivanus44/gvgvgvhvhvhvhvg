import pandas as pd
df = pd.read_csv('heart_disease_health.csv')

df.drop(['HighBP','HighChol','CholCheck','BMI','Smoker','Sex','Stroke','Age','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Education','Income'],axis = 1, inplace = True)
print('У людей с диабетом инфаркт случается чаще')
def fill_int(Age):
    if Age == 1.0:
        return 1
    return 0
df['Diabetes'] = df['Diabetes'].apply(fill_int)
def fill_HDA(HDA):
    if HDA == 1.0:
        return 1
    return 0
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].apply(fill_HDA)
print(df)
print(df['Diabetes'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


X = df.drop('HeartDiseaseorAttack', axis = 1)
y = df['HeartDiseaseorAttack']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('Процент правильного предсказания',accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt

s = pd.Series(data = [249049,4631], index = ['БЕЗ ДИАБЕТА','C ДИАБЕТОМ'])
s.plot(kind = 'barh')

plt.show()
