import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Importando data
# https://cdn.freecodecamp.org/project-data/books/book-crossings.zipn
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Convertendo para dataframe
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# Mesclando os dois dataframes
df = df_ratings.merge(df_books, on='isbn', how='left')

#Visualização do shape do dataframe
df.shape

#Criando um novo dataframe excluindo usuarios e ratings irrelevantes
user_counts = df.groupby('user').size()
book_counts = df.groupby('isbn').size()

popular_users = user_counts[user_counts >= 200].index
popular_books = book_counts[book_counts >= 100].index

filtered = df[(df['user'].isin(popular_users)) & (df['isbn'].isin(popular_books))].index
new_df = df.loc[filtered]

#Visualização do shape do novo dataframe
new_df.shape

#Excluindo duplicações
df = new_df.drop_duplicates(['title', 'user'])

#Verificando nulos
print(df.shape)
df.isna().sum()

#Reestruturando o dataframe
df_pivot = df.pivot(index='title', columns='user', values='rating').fillna(0)

#Visualização do dataframe reestruturado
df_pivot.head()

#Transformando em matrix 2D
df_matrix = csr_matrix(df_pivot.values)

#Criando o modelo NearestNeighbor e alimentando
nearestN = NearestNeighbors(metric = 'cosine', algorithm='brute')
nearestN.fit(df_matrix)

# Função para retornar os livros recomendados
def get_recommends(book = ""):
  recommended_books = [book, []]
  distance, book_info = nearestN.kneighbors([df_pivot.loc[book]], 6, return_distance=True)
  recom_book_info = df_pivot.iloc[np.flip(book_info[0])[:-1]].index.to_list()
  recom_distance = list(np.flip(distance[0])[:-1])

    
  for r in zip(recom_book_info,recom_distance):
      recommended_books[1].append(list(r))

  return recommended_books

#Teste
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge!")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()



#Grafico Clustering
# Aplicar redução de dimensionalidade usando PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_matrix.toarray())

# Aplicar K-Means para agrupar os livros em clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(df_pca)

# Plotar o gráfico de dispersão com cores para grupos de livros
plt.figure(figsize=(10, 8))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('Gráfico de Dispersão com Cores para Grupos de Livros')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()
