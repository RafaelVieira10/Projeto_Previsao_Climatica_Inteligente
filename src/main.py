# src/main.py

import argparse
from data_prep import load_and_preprocess_data
from train import train_model
from evaluate import evaluate
import os

def run_pipeline(random_seed=42):
    """
    Executa o pipeline completo:
    1. Prepara os dados
    2. Treina o modelo
    3. Avalia e gera resultados
    """
    print("--- 1. INÍCIO DO PIPELINE: PREVISÃO DE TEMPERATURA ---")
    
    # 1. Preparação dos Dados
    print("\n--- 2. PREPARAÇÃO DE DADOS (src/data_prep.py) ---")

    # CORREÇÃO AQUI: Removemos o argumento file_path, pois ele é resolvido internamente.
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(random_seed=random_seed)
    
    # 2. Treinamento do Modelo
    print("\n--- 3. TREINAMENTO DO MODELO (src/train.py) ---")
    # O modelo de regressão não precisa de um caminho de modelo fixo, mas mantemos para consistência
    train_model(X_train, y_train) 
    
    # 3. Avaliação e Resultados
    print("\n--- 4. AVALIAÇÃO DO MODELO (src/evaluate.py) ---")
    rmse, predictions = evaluate(X_test, y_test)
    
    # Exemplo de Previsão Simples para a primeira amostra de teste
    print("\n--- 5. PREVISÃO DE EXEMPLO ---")
    if not y_test.empty:
        primeira_previsao = predictions[0]
        temperatura_real = y_test.iloc[0]
        print(f"Temperatura Real (1ª amostra): {temperatura_real:.2f} °C")
        print(f"Temperatura Prevista (1ª amostra): {primeira_previsao:.2f} °C")
    
    print("\n--- 6. PIPELINE CONCLUÍDO COM SUCESSO! ---")
    print(f"Métrica Final: RMSE = {rmse:.2f} °C")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline completo de ML para Previsão de Temperatura.")
    parser.add_argument('--seed', type=int, default=42, help='Semente aleatória fixa para reprodutibilidade.')
    args = parser.parse_args()
    
    run_pipeline(random_seed=args.seed)