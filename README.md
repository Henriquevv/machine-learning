
- [Linear Regression](#Linear-Regression) | [K-NN Algorithm](#K-NN-Algorithm)


# Linear Regression

In this challenge, you will predict healthcare costs using a regression algorithm.

You are given a dataset that contains information about different people including their healthcare costs. Use the data to predict healthcare costs based on new data.

The first two cells of this notebook import libraries and the data.

Make sure to convert categorical data to numbers. Use 80% of the data as the train_dataset and 20% of the data as the test_dataset.

pop off the "expenses" column from these datasets to create new datasets called train_labels and test_labels. Use these labels when training your model.

Create a model and train it with the train_dataset. Run the final cell in this notebook to check your model. The final cell will use the unseen test_dataset to check how well the model generalizes.

To pass the challenge, model.evaluate must return a Mean Absolute Error of under 3500. This means it predicts health care costs correctly within $3500.

The final cell will also predict expenses using the test_dataset and graph the results.




### Demonstração de Código

- Importando dados:  
```
dataset.tail()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/63254540-b438-48f4-9c02-22bcc9b53b5a)


- Dataset após conversão dos Categorical Data em Numeric Data:  
```
dataset.head()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/7cd347a8-a4ca-47b7-81e6-d7d90e508c28)


- Tamanho do ```train_dataset``` e ```test_dataset```. Visualização do ```train_dataset```:  
```
print(len(train_dataset))
print(len(test_dataset))
train_dataset.head()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/b7e5ca9e-da52-49da-a450-037463bc1dbf)


- Visualização da coluna ```expenses``` do ```train_dataset```:  
```
train_labels.head()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/096c26a5-7c15-41df-b98f-a5e45d09303c)


- Visualização do modelo e suas camadas:  
```
model.build()
model.summary()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/fda193dc-0893-4805-908b-ecea89f3f545)


- Resultado Final após alimentação do modelo:  
```
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/9bb3ee4c-e25c-4e80-b991-6db477ffe7bc)


# K-NN Algorithm

In this challenge, you will create a book recommendation algorithm using K-Nearest Neighbors.

You will use the Book-Crossings dataset. This dataset contains 1.1 million ratings (scale of 1-10) of 270,000 books by 90,000 users.

After importing and cleaning the data, use NearestNeighbors from sklearn.neighbors to develop a model that shows books that are similar to a given book. The Nearest Neighbors algorithm measures the distance to determine the “closeness” of instances.

Create a function named get_recommends that takes a book title (from the dataset) as an argument and returns a list of 5 similar books with their distances from the book argument.

Notice that the data returned from get_recommends() is a list. The first element in the list is the book title passed into the function. The second element in the list is a list of five more lists. Each of the five lists contains a recommended book and the distance from the recommended book to the book passed into the function.

If you graph the dataset (optional), you will notice that most books are not rated frequently. To ensure statistical significance, remove from the dataset users with less than 200 ratings and books with less than 100 ratings.




### Demonstração de Código:

- Shape dos dados iniciais:  
```
df.shape
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/184df836-5a3d-455b-b84a-e36a0f3745f3)  

- Shape dos dados após limpeza:  
```
new_df.shape
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/c000016b-984e-4a23-93f2-3d42c858d63f)

- Reestruração dos dados:  
```
df_pivot.head()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/725d81a9-0d53-47c5-8207-6d19655f6cf7)

- Teste do modelo:
```
test_book_recommendation()
```

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/cb3f0b02-874c-4061-803f-d0c73d155ff2)

- Gráfico de Clustering

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/939579d6-3cea-41cb-b89f-2247ef4a2217)

