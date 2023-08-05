import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# Importar dataset
mnist = fetch_openml("mnist_784")

# Convertimos el conjunto de datos en un Dataframe de pandas
df = pd.DataFrame(mnist.data)
print(df)

# Visualizamos los datos
plt.figure(figsize=(20, 4))

for index, digit in zip(range(1, 9), mnist.data[:8]):
    plt.subplot(1, 8, index)
    plt.imshow(np.reshape(digit, (28,28)), cmap=plt.cm.gray)
    plt.title("Ejemplo: " + str(index))

plt.show()

# Dividimos el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.15)

print(len(X_train))
print(len(X_test))

# Entrenamos el algoritmo
clf = MLPClassifier(hidden_layer_sizes=(10,), activation="logistic", solver="sgd")
clf.fit(X_train, y_train)

# Número de capas del perceptrón multicapa
clf.n_layers_

# Número de outputs del perceptrón multicapa
clf.n_outputs_

# 784 * 10 + 10

# Número de parámetros del modelo
clf.coefs_[0].shape

# Dimensiones de la primera capa (hidden layer)
clf.coefs_[1]

# Dimensiones de la segunda capa (output layer)
clf.coefs_[1].shape

# Parámetros bias/intercept que forman parte de cada capa de la red neuronal
clf.intercepts_[0].shape

# Predicción del conjunto de pruebas
y_pred = clf.predict(X_train)

# Mostramos el f1_score resultante de laq clasificación
f1_score(y_test, y_pred, average="weighted")

# Visualizamos las imágenes mal clasificadas
index = 0
index_errors = []

for label, predict in zip(y_test, y_pred):
    if label != predict:
        index_errors.append(index)
    index += 1

plt.figure(figsize=(20, 4))

for i, img_index in zip(range(1, 9), index_errors[8:16]):
    plt.subplot(1, 8, i)
    plt.imshow(np.reshape(X_test[img_index], (28,28)), cmap=plt.cm.gray)
    plt.title("Orig:" + str(y_test[img_index]) + "Pred:" + str(y_pred[img_index]))

plt.show()