# Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import sys

# Dicionário de Mapeamento de Nomes (Inglês -> Português)
COL_MAPPING = {
    'Temperature (C)': 'Temp_Real', 
    'Apparent Temperature (C)': 'Temp_Aparente', 
    'Humidity': 'Umidade', 
    'Wind Speed (km/h)': 'Velocidade_Vento', 
    'Pressure (millibars)': 'Pressao_Atmosferica',
}

def load_and_preprocess_data(random_seed=42):
    """
    Carrega o dataset, renomeia as colunas para português, pré-processa,
    salva a versão processada e divide os dados em treino e teste.
    """
    

    # 1. Obtém o diretório do script atual (Ex: /caminho/para/PROJETO_IA/src)
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    # 2. Sobe um nível (para a raiz do projeto)
    project_root = os.path.dirname(base_dir)
    
    # 3. Constrói o caminho para o arquivo de dados
    file_path = os.path.join(project_root, 'data', 'raw', 'temperature_dataset.csv')
    
    # 4. Constrói o caminho para salvar os dados processados
    processed_dir = os.path.join(project_root, 'data', 'processed')
    processed_file_path = os.path.join(processed_dir, 'processed_temp_data.csv')
    os.makedirs(processed_dir, exist_ok=True) 
    # ====================================================================

    # Mensagem de alerta caso não encontre o arquivo
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Arquivo não encontrado em {file_path}. "
            "Certifique-se de que o 'temperature_dataset.csv' está em 'data/raw/' "
            "e que a estrutura de pastas está correta."
        )

    data = pd.read_csv(file_path)

    # Renomeia as colunas
    data = data.rename(columns=COL_MAPPING)

    # 1. Limpeza
    required_cols = list(COL_MAPPING.values())
    data = data.dropna(subset=required_cols)
    
    # 2. Definir Features (X) e Target (Y)
    target = 'Temp_Real' 
    features = [
        'Temp_Aparente', 
        'Umidade', 
        'Velocidade_Vento', 
        'Pressao_Atmosferica',
    ]
    
    X = data[features]
    y = data[target]

    # 3. Normalização (Escalamento)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
    # 4. SALVAR DADOS PROCESSADOS
    processed_data = X_scaled_df.copy()
    processed_data[target] = y.values
    
    processed_data.to_csv(processed_file_path, index=False)
    print(f"Dados processados (escalados e renomeados) salvos em: {processed_file_path}")
    
    # 5. Divisão Treino/Teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=random_seed
    )

    print(f"Dados prontos para Treino: Treino={len(X_train)} amostras, Teste={len(X_test)} amostras.")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == '__main__':
    # Bloco de teste
    try:
        load_and_preprocess_data()
        print("\nPré-processamento concluído com sucesso.")
    except Exception as e:
        print(f"Erro durante o pré-processamento: {e}")