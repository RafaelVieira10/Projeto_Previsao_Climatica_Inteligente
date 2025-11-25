# PREVISÃO CLIMÁTICA INTELIGENTE (PCI)

## Equipe
- Rafael Vieira Dos Santos — RA: 2224109067
- Ederson Silva Ribeiro Mota — RA: 2224105760
- Turma: Turma 41 - SA | Curso: Ciência da Computação | Período: Noturno | Ano: 2025

## Problema
O objetivo deste projeto é prever a temperatura máxima diária com base em variáveis climáticas históricas (como umidade e pressão atmosférica). A previsão precisa é essencial para o planejamento logístico e econômico em setores sensíveis ao clima, visando minimizar o erro de estimativa.

## Abordagem de IA
Utilizamos um modelo de **Regressão Linear Múltipla**, que é adequado para prever um valor contínuo (temperatura). Este modelo é rápido e transparente, cumprindo o requisito de IA funcional simples. A métrica principal de desempenho é o **Erro Quadrático Médio da Raiz (RMSE)**.

## Dados
**Origem:** Kaggle - https://www.kaggle.com/datasets/budincsevity/szeged-weather
**Esquema:**
- `Data` (Datetime): Data da observação.
- `Temp_Dia_Anterior` (Float): Temperatura do dia anterior.
- `Umidade_Relativa` (Float): Umidade percentual.
- `Pressao_Atmosferica` (Float): Pressão em hPa.
- `Temperatura_Max_Real` (Float): Variável Target (°C).

## Como reproduzir

Siga os passos abaixo para instalar e executar o projeto.

```bash
# 1. Clone o repositório
git clone https://github.com/RafaelVieira10/Projeto_Previsao_Climatica_Inteligente.git
cd Projeto_Previsao_Climatica_Inteligente
code . # Abrir projeto no Visual Studio Code

# 2. Criar e ative um ambiente virtual (recomendado)
# Windows
python -m venv .venv
.\.venv\Scripts\activate
   # OBS: Caso apareça erro de não poder executar scripts, rodar o seguinte comando no powershell, modo elevado (Administrador):
    Set-ExecutionPolicy Unrestricted - # Habilitar scripts no PowerShell
    # A ou S para confirmar a ativação da execução de Scripts.
# Linux/macOS
source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Executar o pipeline completo (pré-processamento, treino, avaliação)
# O script irá: 
# a) processar os dados em data/raw/
# b) treinar o modelo e salvar em models/
# c) calcular métricas, gerar gráficos e salvar em reports/
python src/main.py --seed 42


# Em seguida, executar o exploratory.ipynb, para ánalise dos dados
    # caminho: notebooks\exploratory.ipynb

# 5. Abrir gráficos gerados:
cd reports\figures
explorer . # - Abrir diretório dos gráficos no Windows Explorer 
code .  # - Abrir diretório dos gráficos no Visual Studio Code 
