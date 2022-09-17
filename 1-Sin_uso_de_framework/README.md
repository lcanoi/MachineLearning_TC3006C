## Descripción
1. Crea un repositorio de GitHub para este proyecto.
2. Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) sin usar ninguna biblioteca o framework de aprendizaje máquina, ni de estadística avanzada. Lo que se busca es que implementes manualmente el algoritmo, no que importes un algoritmo ya implementado. 
3. Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.
4. Tu implementación debe de poder correr por separado solamente con un compilador, no debe de depender de un IDE o de un “notebook”. Por ejemplo, si programas en Python, tu implementación final se espera que esté en un archivo .py no en un Jupyter Notebook.
5. Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.

## Librerías utilizadas
- import pandas as pd
- import numpy as np
- from sklearn.model_selection import train_test_split

## Dataset usado
- iris.data
  - (https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

## Métrica de desempeño
- Accuracy = 0.9866
  - (predicciones acertadas / total de predicciones)
  - Con otros random_state alcanzamos Test Accuracy de 1.0 (18, 26, 42)

## Predicciones de prueba

1 )  [5.1, 3.5, 1.4, 0.2]
Predicción:  Iris-setosa
Esperado: Iris-setosa
2 )  [5.0, 3.3, 1.4, 0.2]
Predicción:  Iris-setosa
Esperado: Iris-setosa
3 )  [5.5, 2.3, 4.0, 1.3]
Predicción:  Iris-versicolor
Esperado: Iris-versicolor
4 )  [6.2, 2.9, 4.3, 1.3]
Predicción:  Iris-versicolor
Esperado: Iris-versicolor
5 )  [6.3, 2.8, 5.1, 1.5]
Predicción:  Iris-virginica
Esperado: Iris-virginica

## Archivo a revisar
- NoFramework_ML.py
