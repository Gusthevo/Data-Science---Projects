import pandas as pd
import os

def convert_tsv_to_csv(tsv_file, csv_file):
    """
    Converte arquivo TSV para CSV
    """
    # Lê o TSV
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
    
    # Salva como CSV
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Convertido: {tsv_file} → {csv_file}")

# Exemplo de uso para UM arquivo específico
if __name__ == "__main__":
    # Caminhos para um arquivo específico
    tsv_file = "../data/raw/movie_data/title_basics.tsv"  # ajuste o nome do arquivo
    csv_file = "../data/processed/title.csv"       # ajuste o nome de saída
    
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    convert_tsv_to_csv(tsv_file, csv_file)