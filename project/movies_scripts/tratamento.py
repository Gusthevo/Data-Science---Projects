import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == "__main__":
    df_movies = pd.read_csv("../../data/movies/raw/movies.csv")
    df_ratings = pd.read_csv("../../data/movies/raw/ratings.csv")

    Dataset = pd.merge(df_movies, df_ratings[['tconst', 'averageRating', 'numVotes']], on='tconst', how='left')

    Dataset.drop('tconst', axis=1, inplace=True)

    Dataset = pd.get_dummies(Dataset, columns=['titleType'], prefix='', prefix_sep='')
    Dataset.drop('primaryTitle', axis=1, inplace=True)

    Dataset['genres'] = Dataset['genres'].fillna('\\N')
    Dataset['genres'] = Dataset['genres'].apply(lambda x: [] if x == '\\N' else x.split(','))
    mlb = MultiLabelBinarizer()
    generos_bin = pd.DataFrame(
        mlb.fit_transform(Dataset['genres']),
        columns=mlb.classes_,
        index=Dataset.index
    ).astype(bool)
    Dataset = pd.concat([Dataset.drop(columns=['genres']), generos_bin], axis=1)

    Dataset['startYear'] = Dataset['startYear'].replace('\\N', pd.NA)
    Dataset['startYear'] = Dataset['startYear'].astype('Int64')
    min_ano = Dataset['startYear'].min(skipna=True)
    max_ano = Dataset['startYear'].max(skipna=True)
    bins = list(range(min_ano - min_ano % 5, max_ano + 5, 5))
    labels = [f'{b}-{b+4}' for b in bins[:-1]]
    Dataset['yearInterval'] = pd.cut(Dataset['startYear'], bins=bins, labels=labels, right=True)
    Dataset['yearInterval'] = Dataset['yearInterval'].cat.add_categories(['YearUnknown'])
    Dataset['yearInterval'] = Dataset['yearInterval'].fillna('YearUnknown')
    year_dummies = pd.get_dummies(Dataset['yearInterval'], prefix='Year')
    Dataset = pd.concat([Dataset.drop(columns=['yearInterval']), year_dummies], axis=1)
    Dataset = pd.concat([Dataset.drop(columns=['startYear']), year_dummies], axis=1)
    Dataset.drop('endYear', axis=1, inplace=True)

    Dataset['isAdult'] = Dataset['isAdult'].apply(lambda x: False if x == 0 else True)

    Dataset['runtimeMinutes'] = pd.to_numeric(Dataset['runtimeMinutes'], errors='coerce')
    Dataset['runtimeMinutes'] = Dataset['runtimeMinutes'].round(3)

    bins = [0, 60, 120, 180, np.inf]
    labels = ['Curto', 'Médio', 'Longo', 'Extenso']
    Dataset['runtimeCategory'] = pd.cut(Dataset['runtimeMinutes'], bins=bins, labels=labels)
    Dataset.drop('runtimeMinutes', axis=1, inplace=True)
    Dataset.drop('runtimeCategory', axis=1, inplace=True)

    Dataset = Dataset.dropna(subset=['averageRating'])

    Dataset.to_csv('../../data/movies/processed/Dataset.csv', index=False)