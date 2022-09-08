import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# Input y Output de Data
irisColumns = ['f1','f2','f3','f4','output']
df = pd.read_csv('data/iris.data', names=irisColumns)

# Cambiar output de especies a números [0,1,2]
factor = pd.factorize(df['output'])
df.output = factor[0]
definitions = factor[1]

input = df[['f1','f2','f3','f4']].values
output = df['output'].values

x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.3, random_state = 42)
# Notas: random_state = 11 da acc baja, random_state = 42 da acc perfecta, esto con los modelos en random_state = 18

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models = []
models.append(('K-Neighbors', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('SVC', SVC(random_state = 18)))
models.append(('Regresion Logistica', LogisticRegression(random_state = 18)))
models.append(('Arboles Desicion', DecisionTreeClassifier(random_state = 18)))
models.append(('Multilayer Perceptron', MLPClassifier(random_state = 18)))
models.append(('Bosques Aleatorios g', RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 18)))
models.append(('Bosques Aleatorios e', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 18)))

nom = [] 
acc = [] 
acc2 = []
xtab = []

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc.append(accuracy_score(y_test, y_pred))
    acc2.append(model.score(x_test, y_test))
    nom.append(name)

    #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(3),definitions))
    y_testv = np.vectorize(reversefactor.get)(y_test)
    y_predv = np.vectorize(reversefactor.get)(y_pred)
    # # Making the Confusion Matrix
    xtab.append(pd.crosstab(y_testv, y_predv, rownames=['Actual Species'], colnames=['Predicted Species']))

tr_split = pd.DataFrame({'Nombre': nom, 'Accuracy': acc, 'A2': acc2})
print(tr_split, "\n")

for i in range(len(xtab)):
    print(nom[i], ":")
    print(xtab[i], "\n")


'''  una regresión líneal o logística y que empiece con un par de características. '''


''' Entrena sobre un dataset '''


''' Muestra la métrica de desempeño del modelo '''


''' Corre predicciones para validar salida del modelo '''

