import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
Modelo: Regresión Logística Multiclase
Entradas: petal length, petal width, sepal length, sepal width
Salidas: 0, 1, 2 (iris-setosa, iris-versicolor, iris-virginica)
Hipótesis: z = wx + b
Activación: sigmoid(z)
'''

# Input y Output de Data
irisColumns = ['f1','f2','f3','f4','output']
df = pd.read_csv('./data/iris.data', names=irisColumns)

# Cambiar output de especies a números [0,1,2]
fac = pd.factorize(df['output'])
df.output = fac[0] # fac[0] es el array de todos los outputs como 0s, 1s y 2s
species = fac[1] # fac[1] es el Index, los nombres de las especies

# Dividir datos en input y output
input = df[['f1','f2','f3','f4']].values
output = df['output'].values

# Dividir datos en train y test
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.5, random_state = 32)
# random_state 18, 26, 42 accuracy = 1.0
# random_state 28 accuracy = 0.893
# validación cruzada es necesaria para este problema
# implementación de esto lo vemos en Framework_ML.py

variables = len(df.columns) - 1 # 4
classes = len(df['output'].unique()) # 3
t_rows = len(x_train) # 120 (con 0.2 test_size)

# Función de activación
def sigmoid(x):
    return 1/(1+np.exp(-x))

'''
Con sigmoide, si z va al infinito, la salida es 1, si z va a -infinito, la salida es 0,
pero nosotros queremos 3 clases...

Entonces se crea un arreglo onehot representando con 1s y 0s las 3 clases a las que 
pertenece o no cada entrada
Así pues si una entrada pertenece a la primer clase (iris-setosa), el arreglo de onehot 
representaría el output como [1,0,0]
Asimismo se crea un arreglo de weights con las dimensiones de (variables, clases) y bias
con dimensiones (1, clases) ya que en el cálculo se va a realizar la activación de cada 
una de las 3 clases para cada variable (sólo una de las 3 debería dar resultado = 1, las 
otras 2 deberían dar 0)
'''

onehot = np.zeros((t_rows, classes))
for i in range(t_rows):
    onehot[i, y_train[i]] = 1

weights = np.zeros((variables, classes))

bias = np.zeros((1, classes))

lr = 10e-3
epochs = 1000

print("Starting Vaues\n______________________\n")
print("Weights:\n", weights)
print("Bias:\n", bias)
print("Learning Rate: ", lr)
print("Epochs: ", epochs, "\n\n")

# Entrenar modelo
for epoch in range(epochs):
    for i, x in enumerate(x_train):
        # Forward Propagation
        # Calcular activación
        z = np.dot(x, weights) + bias
        a = sigmoid(z)

        # Backward Propagation
        # Calcular derivativas de gradientes
        error = a - onehot[i]
        d_grad_w = np.dot(x.reshape(-1,1), error.reshape(1,-1))
        d_grad_b = error

        # Actualizar pesos y bias
        weights -= lr * d_grad_w
        bias -= lr * d_grad_b

print("Final Values\n______________________\n")
print("Weights:\n", weights)
print("Bias:\n", bias, "\n\n")

# Predicciones puntuales
p1 = [5.1,3.5,1.4,0.2] # Iris-setosa
p2 = [5.0,3.3,1.4,0.2] # Iris-setosa
p3 = [5.5,2.3,4.0,1.3] # Iris-versicolor
p4 = [6.2,2.9,4.3,1.3] # Iris-versicolor
p5 = [6.3,2.8,5.1,1.5] # Iris-virginica
pArr = [p1, p2, p3, p4, p5]

print("5 punctual predictions:")
for i in range(len(pArr)):
    z = np.dot(pArr[i], weights) + bias
    a = sigmoid(z)
    print(i+1, ") ", pArr[i])
    print("Prediction: ", species[np.argmax(a)])
    # Valores esperados para los 5 puntos:
    if i <= 1:
        print("Expected: Iris-setosa")
    elif i <= 3:
        print("Expected: Iris-versicolor")
    else:
        print("Expected: Iris-virginica")

# Predicciones de datos test
y_pred = []
for i in range(len(x_test)):
    z = np.dot(x_test[i], weights) + bias
    a = sigmoid(z)
    y_pred.append(np.argmax(a))

# La métrica de desempeño del modelo es la accuracy
# predicciones acertadas / total de predicciones
print("\n\n Final Test Accuracy: ", np.sum(y_pred == y_test)/len(y_test), "\n\n")

# Convertimos de vuelta los números a las especies
revFactor = dict(zip(range(classes),species))
y_test_rf = np.vectorize(revFactor.get)(y_test)
y_pred_rf = np.vectorize(revFactor.get)(y_pred)

# Matriz de confusión
print("Confusion Matrix\n________________\n")
print(pd.crosstab(y_test_rf, y_pred_rf, rownames=['Real'], colnames=['Predictions']))
