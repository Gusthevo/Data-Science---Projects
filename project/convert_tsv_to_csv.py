import pandas as pd
import os

def convert_tsv_to_csv(tsv_file, csv_file):
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
    
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Convertido: {tsv_file} → {csv_file}")

if __name__ == "__main__":
    tsv_file = "../data/movies/title.ratings.tsv" 
    csv_file = "../data/movies/ratings.csv"       
    
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    convert_tsv_to_csv(tsv_file, csv_file)