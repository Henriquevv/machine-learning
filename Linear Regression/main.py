import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


#Importando data
dataset = pd.read_csv('C:/Users/henri/OneDrive/Área de Trabalho/ufs/projetos/Machine Learning/TensorFlow/Linear Regression/insurance.csv')
dataset.tail()

#Convertendo Categorical Data em Numeric Data
df = dataset
df["sex"] = pd.factorize(df["sex"])[0]
df["region"] = pd.factorize(df["region"])[0]
df["smoker"] = pd.factorize(df["smoker"])[0]
dataset = df
dataset.head()

#Randomizando as posições das informações e seperando em train_dataset e test_dataset (80% e 20%)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

train_dataset = dataset[0:int(0.8 * dataset.shape[0])]
test_dataset = dataset[int(0.8 * dataset.shape[0]):]

print(len(train_dataset))
print(len(test_dataset))
train_dataset.head()

#Separando a coluna "expenses"
train_labels = train_dataset.pop("expenses")
test_labels = test_dataset.pop("expenses")

train_labels.head()

#Preparando o modelo
normalizer = keras.layers.Normalization()
normalizer.adapt(np.array(train_dataset))

model = keras.Sequential([
    normalizer,
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dropout(.2),
    keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)
model.build()
model.summary()

#Treinando o modelo
history = model.fit(
    train_dataset,
    train_labels,
    epochs = 100,
    verbose = 0
)

print(history)



# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)