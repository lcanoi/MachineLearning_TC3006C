## Descripción
1. Crea un repositorio de GitHub para este proyecto.
2. Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) haciendo uso de una biblioteca o framework de aprendizaje máquina. Lo que se busca es que demuestres tu conocimiento sobre el framework y como configurar el algoritmo. 
3. Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.
4. Tu implementación debe de poder correr por separado solamente con un compilador, no debe de depender de un IDE o de un “notebook”. Por ejemplo, si programas en Python, tu implementación final se espera que esté en un archivo .py no en un Jupyter Notebook.
5. Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.

## Librerías utilizadas
- import pandas as pd
- import numpy as np
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import accuracy_score

- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier

## Dataset usado
- iris.data
  - (https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

## Métrica de desempeño
- accuracy_score
  - (Accuracy classification score. This function computes subset accuracy)

## Predicciones de prueba

| Predicciones    | Iris-setosa | Iris-visecolor | Iris-virginica |
|-----------------|-------------|----------------|----------------|
| Reales          |             |                |                |
| Iris-setosa     | 29          | 0              | 0              |
| Iris-versicolor | 0           | 23             | 0              |
| Iris-virginica  | 0           | 0              | 23             |

Con test_size = 0.5 y random_state = 42 alcanzamos accuracy de 1.0 en 
modelos de árboles.

## Archivo a revisar
- NoFramework_ML.py
