import pandas as pd
df = pd.read_csv('heart_disease_health.csv')
print('У людей с ожирением чаще случается инфаркт')
print(df)
df.drop(['HighBP','HighChol','CholCheck','Smoker','Sex','Stroke','Diabetes','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Age','Education','Income'],axis = 1, inplace = True)
def fill_int(BMI):
    if BMI >= 25:
        return 1
    return 0
df['BMI'] = df['BMI'].apply(fill_int)
def fill_HDA(HDA):
    if HDA == 1.0:
        return 1
    return 0
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].apply(fill_HDA)
print(df)

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

s = pd.Series(data = [212,186], index = ['С ожирением','Без ожирения'])
s.plot(kind = 'barh')

plt.show()
