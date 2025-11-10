# %% [markdown]
# # **Configurando**
# Realizando importações das bibliotecas utilizadas durante a análise exploratória 

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# %% [markdown]
# Lendo Datasets de Movies e Ratings

# %%
df_movies = pd.read_csv("../../data/movies/movies.csv")
df_ratings = pd.read_csv("../../data/movies/ratings.csv")

# %%
print("Movies Dataset:\n")
print(df_movies.head())
print("Ratings Dataset:\n")
print(df_ratings)

# %% [markdown]
# # **Estruturando em um único Dataset**
# Adicionado ao dataset Movies as features averageRating e numVotes

# %%
Dataset = pd.merge(df_movies, df_ratings[['tconst', 'averageRating', 'numVotes']], on='tconst', how='left')
print(Dataset.head())
# Quantidade de NaN por coluna
nan_count = Dataset.isna().sum()
print(nan_count)


# %% [markdown]
# # **Explorando Features do Dataset**
# 
# Vendo os tipos de cada feature

# %%
print(Dataset.dtypes)

# %% [markdown]
# Eliminando tconst por se tratar apenas de id do IMDB

# %%
Dataset.drop('tconst', axis=1, inplace=True)
print(Dataset)

# %% [markdown]
# Analisando o que tem em cada coluna

# %%
for col in Dataset.columns:
    print(f"\nColuna: {col}")
    print(Dataset[col].unique())


# %% [markdown]
# Subdividir a feature categorica 'titleType' em várias features binarias atráves do One Hoting Encode

# %%
Dataset = pd.get_dummies(Dataset, columns=['titleType'], prefix='', prefix_sep='')
print(Dataset.dtypes)

# %% [markdown]
# Checando conteúdo dos titulos

# %%
total = len(Dataset)
dif = Dataset['primaryTitle'] != Dataset['originalTitle']
dif_count = dif.sum()
print(f"Diferentes: {dif_count} de {total} ({dif_count/total:.2%})")
diferentes = Dataset[dif]
print(diferentes[['primaryTitle', 'originalTitle']])


# %% [markdown]
# São só nomes populares e os nomes originais do filme, pode ser retirado.
# Então vamos retirar a coluna 'primaryTitle'

# %%
Dataset.drop('primaryTitle', axis=1, inplace=True)
print(Dataset.head())

# %% [markdown]
# Usando One-Hot Encoding para transformar os generos em features boleanas

# %%
Dataset['genres'] = Dataset['genres'].fillna('\\N')

# Transforma em listas (quem for \N vira lista vazia)
Dataset['genres'] = Dataset['genres'].apply(lambda x: [] if x == '\\N' else x.split(','))

# Aplica o MultiLabelBinarizer
mlb = MultiLabelBinarizer()
generos_bin = pd.DataFrame(
    mlb.fit_transform(Dataset['genres']),
    columns=mlb.classes_,
    index=Dataset.index
).astype(bool)

# Junta e remove a coluna original
Dataset = pd.concat([Dataset.drop(columns=['genres']), generos_bin], axis=1)

print(Dataset.head())
print(f"\nTotal de linhas: {len(Dataset)}")

# %%
print(Dataset.dtypes)

# %%
print(Dataset['startYear'].unique())

# %%
Dataset['startYear'] = Dataset['startYear'].replace('\\N', pd.NA)
Dataset['startYear'] = Dataset['startYear'].astype('Int64')  # permite NaN

# 2️⃣ Define intervalos de 5 anos
min_ano = Dataset['startYear'].min(skipna=True)
max_ano = Dataset['startYear'].max(skipna=True)
bins = list(range(min_ano - min_ano % 5, max_ano + 5, 5))
labels = [f'{b}-{b+4}' for b in bins[:-1]]

# 3️⃣ Cria nova coluna com o intervalo de 5 anos
Dataset['yearInterval'] = pd.cut(Dataset['startYear'], bins=bins, labels=labels, right=True)

# 4️⃣ Substitui NaN por 'YearUnknown'
Dataset['yearInterval'] = Dataset['yearInterval'].cat.add_categories(['YearUnknown'])
Dataset['yearInterval'] = Dataset['yearInterval'].fillna('YearUnknown')

# 5️⃣ One-hot encoding
year_dummies = pd.get_dummies(Dataset['yearInterval'], prefix='Year')

# 6️⃣ Junta ao dataset e remove coluna original
Dataset = pd.concat([Dataset.drop(columns=['yearInterval']), year_dummies], axis=1)
Dataset = pd.concat([Dataset.drop(columns=['startYear']), year_dummies], axis=1)

# 7️⃣ Resultado
print(Dataset)

# %% [markdown]
# EndYear é apenas o ano que terminou a serie de tv como não se aplica a todos os tipos, pode ser descartado

# %%
Dataset.drop('endYear', axis=1, inplace=True)
print(Dataset.head())

# %% [markdown]
# Analisando como está cada coluna

# %%
Dataset['isAdult'] = Dataset['isAdult'].apply(lambda x: False if x == 0 else True)
print(Dataset['isAdult'].unique())


# %% [markdown]
# Padronzando runtime minutes para int

# %%
Dataset['runtimeMinutes'] = pd.to_numeric(Dataset['runtimeMinutes'], errors='coerce')
Dataset['runtimeMinutes'] = Dataset['runtimeMinutes'].round(3)
print(Dataset['runtimeMinutes'].head())

# %%
bins = [0, 60, 120, 180, np.inf]
labels = ['Curto', 'Médio', 'Longo', 'Extenso']
Dataset['runtimeCategory'] = pd.cut(Dataset['runtimeMinutes'], bins=bins, labels=labels)

# %%
print(Dataset['runtimeCategory'])

# %%
# Quantidade de NaN por coluna
nan_count = Dataset.isna().sum()
print(nan_count)



