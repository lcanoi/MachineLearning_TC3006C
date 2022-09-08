import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Input y Output de Data
irisColumns = ['f1','f2','f3','f4','output']
df = pd.read_csv('./data/iris.data', names=irisColumns)

classes = len(df['output'].unique()) # 3

# Cambiar output de especies a números [0,1,2]
factor = pd.factorize(df['output'])
df.output = factor[0] # Factor[0] es el array de 0s, 1s y 2s
species = factor[1] # Factor[1] es el Index, las especies

# Dividir datos en input y output
input = df[['f1','f2','f3','f4']].values
output = df['output'].values

# Dividir datos en train y test
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.5, random_state = 42)
'''
Es muy facil alcanzar accuracy de 1.0 con el iris dataset, por lo cual
uso un test_size del 50%
Aún así se puede alcanzar accuracy de 1.0, pero es menos probable que el
modelo final esté sobreajustado
'''
# Notas: random_state = 11 da acc baja, random_state = 42 da acc perfecta, esto con los modelos en random_state = 18

# Normalizar datos
# media = 0 y desviación estándar = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Algunos modelos implementados vistos en clase
models = []
models.append(('K-Neighbors', KNeighborsClassifier(n_neighbors = 9)))
models.append(('Logistic Regression', LogisticRegression(solver='newton-cg', max_iter=50, multi_class='multinomial', random_state = 0)))
models.append(('Decision Tree', DecisionTreeClassifier(criterion = 'gini', max_depth = 3, min_samples_leaf = 1, random_state = 0)))
models.append(('Random Forest', RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 42)))

nom = [] 
acc = [] 
xtab = []

# Entrenar modelos
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc.append(accuracy_score(y_test, y_pred)) # También calculable con model.score(x_test, y_test)
    nom.append(name)

    # Convertimos de vuelta los números a las especies
    revFactor = dict(zip(range(classes),species))
    y_test_rf = np.vectorize(revFactor.get)(y_test)
    y_pred_rf = np.vectorize(revFactor.get)(y_pred)
    # Matriz de confusión
    xtab.append(pd.crosstab(y_test_rf, y_pred_rf, rownames=['Real'], colnames=['Predictions']))

# Muestra de la métrica de desempeño de los modelos (sobre el subset de prueba)
tr_split = pd.DataFrame({'Nombre': nom, 'Accuracy': acc})
print("Accuracy of the models with the test subset")
print(tr_split, "\n")

# Muestra de las predicciones de uno de los modelos
# Selecionamos uno de los de mejor desempleño, en este caso Arbol de Decisión
modelNo = 2
print("Predictions of the", models[modelNo][0], "model with the test subset")
print("Confusion Matrix\n________________\n")
print(xtab[modelNo], "\n")

# for i in range(len(xtab)):
#     print(nom[i], ":")
#     print(xtab[i], "\n")