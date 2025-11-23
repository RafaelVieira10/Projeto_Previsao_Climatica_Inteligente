# src/evaluate.py

from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import os

def load_model(model_path='models/modelo_previsao.pkl'):
    """Carrega o modelo treinado."""
    try:
        model = load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Execute src/train.py primeiro.")

def calculate_metrics(y_test, predictions):
    """Calcula e exibe as métricas de Regressão."""
    
    # Métrica Principal: RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    # Métrica Secundária: MAE
    mae = mean_absolute_error(y_test, predictions)
    
    print("\n--- Resultados da Avaliação ---")
    print(f"Métrica Principal (RMSE): {rmse:.2f} °C")
    print(f"MAE: {mae:.2f} °C")
    print("-----------------------------\n")
    return rmse, mae

def generate_plots(y_test, predictions, X_test, save_dir='reports/figures'):
    """Gera e salva os três gráficos de desempenho em português."""
    
    os.makedirs(save_dir, exist_ok=True)
    y_test = y_test.reset_index(drop=True) 
    
    # 1. Gráfico de Dispersão: Real vs. Previsto 
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Temperatura Real (°C)')
    plt.ylabel('Temperatura Prevista (°C)')
    plt.title('1. Dispersão: Temperatura Real vs. Prevista (Teste)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'scatter_real_vs_previsto.png'))
    plt.close()

    # 2. Histograma dos Resíduos 
    residuals = y_test - predictions
    plt.figure(figsize=(7, 6))
    plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', linewidth=1.5)
    plt.xlabel('Resíduos (Erro: Real - Previsto) (°C)')
    plt.ylabel('Frequência')
    plt.title('2. Histograma da Distribuição dos Resíduos')
    plt.savefig(os.path.join(save_dir, 'histograma_residuos.png'))
    plt.close()

    # 3. Série Temporal: Comparação Previsto x Real (últimas 40 amostras) 
    sample_size = min(40, len(y_test))
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[-sample_size:].values, label='Temperatura Real', marker='o', linestyle='-', linewidth=2)
    plt.plot(predictions[-sample_size:], label='Temperatura Prevista', marker='x', linestyle='--', linewidth=1)
    plt.xlabel('Amostras de Teste (Dias)')
    plt.ylabel('Temperatura (°C)')
    plt.title(f'3. Série Temporal: Comparação Real vs. Prevista (Últimos {sample_size} Dias)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'serie_temporal_comparacao.png'))
    plt.close()

    print(f"Três gráficos gerados e salvos em português em: {save_dir}/")
    
def evaluate(X_test, y_test):
    """Pipeline de avaliação do modelo."""
    
    model = load_model()
    predictions = model.predict(X_test)
    
    # Calcular e exibir métricas
    rmse, _ = calculate_metrics(y_test, predictions)
    
    # Gerar e salvar gráficos
    generate_plots(y_test, predictions, X_test)
    
    return rmse, predictions

if __name__ == '__main__':
    # Teste de funcionalidade
    from data_prep import load_and_preprocess_data
    
    if not os.path.exists('models/modelo_previsao.pkl'):
        print("Modelo não encontrado. Execute o src/main.py primeiro.")
    elif not os.path.exists('data/raw/temperature_dataset.csv'):
         print("Dataset não encontrado em data/raw/. Não foi possível testar a avaliação.")
    else:
        # Carrega os dados de teste (sempre com a semente fixa)
        _, X_test, _, y_test, _ = load_and_preprocess_data(random_seed=42)
        evaluate(X_test, y_test)