
- [Linear Regression](#Linear-Regression) |


# Linear Regression

In this challenge, you will predict healthcare costs using a regression algorithm.

You are given a dataset that contains information about different people including their healthcare costs. Use the data to predict healthcare costs based on new data.

The first two cells of this notebook import libraries and the data.

Make sure to convert categorical data to numbers. Use 80% of the data as the train_dataset and 20% of the data as the test_dataset.

pop off the "expenses" column from these datasets to create new datasets called train_labels and test_labels. Use these labels when training your model.

Create a model and train it with the train_dataset. Run the final cell in this notebook to check your model. The final cell will use the unseen test_dataset to check how well the model generalizes.

To pass the challenge, model.evaluate must return a Mean Absolute Error of under 3500. This means it predicts health care costs correctly within $3500.

The final cell will also predict expenses using the test_dataset and graph the results.




## Code Demonstration

```
dataset.tail()
```
Importando dados:  

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/63254540-b438-48f4-9c02-22bcc9b53b5a)


```
dataset.head()
```
Dataset após conversão dos Categorical Data em Numeric Data:  

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/7cd347a8-a4ca-47b7-81e6-d7d90e508c28)


```
print(len(train_dataset))
print(len(test_dataset))
train_dataset.head()
```
Tamanho do ```train_dataset``` e ```test_dataset```. Visualização do ```train_dataset```:  

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/b7e5ca9e-da52-49da-a450-037463bc1dbf)


```
train_labels.head()
```
Visualização da coluna ```expenses``` do ```train_dataset```  

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/096c26a5-7c15-41df-b98f-a5e45d09303c)


```
model.build()
model.summary()
```
Visualização do modelo e suas camadas:  

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/fda193dc-0893-4805-908b-ecea89f3f545)


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
Resultado Final após alimentação do modelo:  

![image](https://github.com/Henriquevv/machine-learning/assets/71598959/9bb3ee4c-e25c-4e80-b991-6db477ffe7bc)
