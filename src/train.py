# Treinamento do modelo (Regressão Linear/SVR)

# src/train.py

from sklearn.linear_model import LinearRegression
from joblib import dump
import os

def train_model(X_train, y_train, model_path='models/modelo_previsao.pkl'):
    """
    Treina o modelo de Regressão Linear e o salva no disco.
    """
    
    # 1. Inicializar e Treinar o Modelo
    print("Iniciando o treinamento do modelo de Regressão Linear...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Treinamento concluído.")
    
    # 2. Salvar o Modelo
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    dump(model, model_path)
    print(f"Modelo salvo em: {model_path}")
    
    return model

if __name__ == '__main__':
    # Este módulo requer que o data_prep.py tenha rodado antes
    from data_prep import load_and_preprocess_data
    
    # Garante que o dataset existe antes de tentar carregar
    if not os.path.exists('data/raw/temperature_dataset.csv'):
        print("Primeiro, execute o main.py ou data_prep.py para gerar os dados.")
    else:
        # Carrega os dados de treino (sempre com a semente fixa)
        X_train, _, y_train, _, _ = load_and_preprocess_data(random_seed=42)
        train_model(X_train, y_train)