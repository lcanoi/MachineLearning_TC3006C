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
factor = pd.factorize(df['output'])
df.output = factor[0] # Factor[0] es el array de 0s, 1s y 2s
species = factor[1] # Factor[1] es el Index, las especies

# Dividir datos en input y output
input = df[['f1','f2','f3','f4']].values
output = df['output'].values

# Dividir datos en train y test
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.2, random_state = 42)

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

# Realizar predicciones de datos test
y_pred = []
for i in range(len(x_test)):
    z = np.dot(x_test[i], weights) + bias
    a = sigmoid(z)
    y_pred.append(np.argmax(a))

# La métrica de desempeño del modelo es la accuracy
# predicciones acertadas / total de predicciones
print("Test Accuracy: ", np.sum(y_pred == y_test)/len(y_test), "\n\n")

# Convertimos de vuelta los números a las especies
revFactor = dict(zip(range(classes),species))
y_test_rf = np.vectorize(revFactor.get)(y_test)
y_pred_rf = np.vectorize(revFactor.get)(y_pred)

# Matriz de confusión
print("Test data predictions:\n")
print("Confusion Matrix\n________________\n")
print(pd.crosstab(y_test_rf, y_pred_rf, rownames=['Real'], colnames=['Predictions']))
