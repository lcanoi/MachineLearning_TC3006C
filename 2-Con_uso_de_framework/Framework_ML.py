import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier


''' /// CARGA Y TRANSFORMACIÓN DE DATOS /// '''

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
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.2, random_state = 42)
'''
Vamos a comenzar con test size del 20%
Esto nos ayudará a hacer una compración más clara cuando realizemos el 
cross validation, ya que haremos 5 folds y cada test size será igualmente
del 20% de los datos
'''

# Normalizar datos
# media = 0 y desviación estándar = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


''' /// CREACIÓN Y ENTRENAMIENTO DEL MODELO /// '''

# Crear modelos
models = []
# models.append(('Decision Tree b', DecisionTreeClassifier(random_state=0)))
# models.append(('Decision Tree Final 1', DecisionTreeClassifier(criterion = 'entropy', max_depth = 6, min_samples_leaf = 1, min_samples_split = 2, random_state = 0)))
models.append(('Decision Tree Final 2', DecisionTreeClassifier(criterion = 'gini', max_depth = 4, min_samples_leaf = 3, min_samples_split = 6, random_state = 0)))

nom = [] 
train_acc = [] 
test_acc = []
preds = []
xtab = []

# Entrenar modelos
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    preds.append(y_pred)
    train_acc.append(accuracy_score(y_train, model.predict(x_train)))
    test_acc.append(accuracy_score(y_test, y_pred))
    nom.append(name)

    # Convertimos de vuelta los números a las especies
    revFactor = dict(zip(range(classes),species))
    y_test_rf = np.vectorize(revFactor.get)(y_test)
    y_pred_rf = np.vectorize(revFactor.get)(y_pred)
    # Matriz de confusión
    xtab.append(pd.crosstab(y_test_rf, y_pred_rf, rownames=['Real'], colnames=['Predictions']))

# Muestra de la métrica de desempeño de los modelos (Precisión)
tr_split = pd.DataFrame({'Nombre': nom, 'Train Accuracy': train_acc, 'Test Accuracy': test_acc})
print("Accuracy of the models with the subsets")
print(tr_split, "\n")

'''
Este progmama está configurado para entrenar a varios modelos a la vez.
Si guardas más de un modelo en el arreglo de models, puedes seleccionar
sobre cual modelo hacer el análisis modificando a modelNo
'''
modelNo = 0


''' /// VISUALIZACIÓN DE RESULTADOS /// '''

# Muestra de las predicciones del modelo entrenado
print("Predictions of the", models[modelNo][0], "model with the test subset")
print("Confusion Matrix\n________________\n")
print(xtab[modelNo], "\n")

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree

# Graficar heatmap de la matriz de confusión de un modelo
# import seaborn as sns
# plt.figure(figsize=(10,7))
# sns.heatmap(xtab[modelNo], annot=True)
# plt.show()

# Grafica final del Decision Tree
plt.figure(figsize=(15,10))
tree.plot_tree(models[modelNo][1], feature_names=irisColumns[:-1], class_names=species, filled=True, rounded=True, fontsize=14)
plt.show()


''' /// VALIDACIÓN DEL MODELO /// '''

# Validación cruzada del modelo
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

# Hacemos la KFold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Las métricas que queremos obtener
scorer = make_scorer(accuracy_score)

# Cross validate
cv_results = cross_validate(models[modelNo][1], input, output, cv=kfold, scoring=scorer, return_train_score=True)

# Imprimimos resultados relevantes
print("Accuracy of the", models[modelNo][0], "model with k-fold cross validation")
print("K-fold test accuracies:\n", cv_results['test_score'])
print("K-fold train accuracies:\n", cv_results['train_score'])
print("Mean test accuracy:", cv_results['test_score'].mean())
print("Mean train accuracy:", cv_results['train_score'].mean())
print("Standard deviation: ", cv_results['test_score'].std())
print("Variance: ", cv_results['test_score'].var())
print("Bias: ", 1 - cv_results['test_score'].mean())


''' /// MEJORAR DESEMPEÑO DEL MODELO /// '''

# Tuning de hiperparámetros
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Grid de hiperparámetros
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 3, 4],
    'min_samples_split': [2, 3, 4, 5, 6],
    'criterion': ['gini', 'entropy'],
}

# Podemos utilizar GridSearchCV o RandomizedSearchCV
# La diferencia es que GridSearchCV prueba todas las combinaciones de hiperparámetros 
# mientras que RandomizedSearchCV prueba una muestra aleatoria de combinaciones
clf = RandomizedSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs = -1, verbose = 0)

# Entrenamos el clasificador
clf.fit(input, output)

# Imprimimos los mejores hiperparámetros
print("Best parameters found by the grid search:")
print(clf.best_params_)

