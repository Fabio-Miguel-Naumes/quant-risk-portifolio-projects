"""
Modelagem de Volatilidade com ARCH/GARCH (Versão Simplificada)
==========================================================

Este script implementa modelos ARCH (Autoregressive Conditional Heteroskedasticity) e 
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) para analisar e 
modelar a volatilidade de séries temporais financeiras.

Versão simplificada para nível básico a intermediário.
"""

# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.stats import norm
import warnings
from datetime import datetime, timedelta

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
warnings.filterwarnings('ignore')

# Configurar seed para reprodutibilidade
np.random.seed(42)

def generate_simulated_returns(n_days=500):
    """
    Gera retornos diários simulados com clusters de volatilidade.
    
    Parâmetros:
    -----------
    n_days : int
        Número de dias a serem simulados
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame com datas e retornos simulados
    """
    # Criar datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_days)
    
    # Simular volatilidade variante no tempo
    daily_vol = np.zeros(n_days)
    daily_vol[0] = 0.01  # Volatilidade diária inicial
    
    # Parâmetros para o processo de volatilidade
    mean_reversion = 0.90
    vol_of_vol = 0.2
    base_vol = 0.01
    
    # Gerar processo de volatilidade
    for t in range(1, n_days):
        # Processo de reversão à média com choque aleatório
        shock = np.random.normal(0, vol_of_vol / np.sqrt(252))
        daily_vol[t] = daily_vol[t-1] * mean_reversion + (1 - mean_reversion) * base_vol + shock
        daily_vol[t] = max(daily_vol[t], 0.001)  # Garantir volatilidade positiva
    
    # Gerar retornos com a volatilidade simulada
    mean_return = 0.0005  # Retorno médio diário
    returns = np.random.normal(mean_return, daily_vol)
    
    # Adicionar alguns eventos extremos (caudas pesadas)
    num_jumps = int(n_days * 0.03)  # 3% dos dias terão saltos
    jump_idx = np.random.choice(range(n_days), num_jumps, replace=False)
    jump_size = np.random.normal(0, 0.03, num_jumps)  # Tamanho dos saltos
    returns[jump_idx] += jump_size
    
    # Criar DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Returns': returns,
        'True_Volatility': daily_vol
    })
    df.set_index('Date', inplace=True)
    
    return df

def plot_returns_and_volatility(returns_df):
    """
    Plota os retornos e a volatilidade verdadeira.
    
    Parâmetros:
    -----------
    returns_df : pd.DataFrame
        DataFrame com retornos e volatilidade
    """
    # Criar figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plotar retornos
    ax1.plot(returns_df.index, returns_df['Returns'], color='blue', alpha=0.7)
    ax1.set_title('Retornos Diários Simulados', fontsize=14)
    ax1.set_ylabel('Retorno', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plotar volatilidade verdadeira
    ax2.plot(returns_df.index, returns_df['True_Volatility'], color='red', alpha=0.7)
    ax2.set_title('Volatilidade Diária Verdadeira', fontsize=14)
    ax2.set_ylabel('Volatilidade', fontsize=12)
    ax2.set_xlabel('Data', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('returns_and_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()

def fit_garch_model(returns):
    """
    Ajusta um modelo GARCH(1,1) aos retornos.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série de retornos
        
    Retorna:
    --------
    modelo ajustado e resultados
    """
    # Criar e ajustar modelo GARCH(1,1)
    # p=1: ordem do componente ARCH
    # q=1: ordem do componente GARCH
    model = arch_model(returns, p=1, q=1, mean='Constant', vol='GARCH', dist='normal')
    results = model.fit(disp='off')
    return model, results

def plot_garch_results(returns_df, garch_results):
    """
    Plota os resultados do modelo GARCH.
    
    Parâmetros:
    -----------
    returns_df : pd.DataFrame
        DataFrame com retornos
    garch_results : ARCHModelResult
        Resultados do modelo GARCH
    """
    # Obter volatilidade condicional estimada pelo modelo
    conditional_vol = garch_results.conditional_volatility
    
    # Criar figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plotar retornos e volatilidade condicional
    ax1.plot(returns_df.index, returns_df['Returns'], color='blue', alpha=0.4, label='Retornos')
    ax1.plot(returns_df.index, conditional_vol, color='red', alpha=0.7, label='Volatilidade GARCH')
    ax1.plot(returns_df.index, -conditional_vol, color='red', alpha=0.7)
    ax1.fill_between(returns_df.index, -conditional_vol, conditional_vol, color='red', alpha=0.1)
    ax1.set_title('Retornos e Volatilidade Condicional GARCH', fontsize=14)
    ax1.legend()
    
    # Plotar comparação entre volatilidade verdadeira e estimada
    ax2.plot(returns_df.index, returns_df['True_Volatility'], color='blue', alpha=0.7, label='Volatilidade Verdadeira')
    ax2.plot(returns_df.index, conditional_vol, color='red', alpha=0.7, label='Volatilidade GARCH')
    ax2.set_title('Comparação: Volatilidade Verdadeira vs. Estimada', fontsize=14)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_ylabel('Volatilidade', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('garch_model_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def forecast_volatility(garch_results, horizon=30):
    """
    Realiza previsão de volatilidade para um horizonte específico.
    
    Parâmetros:
    -----------
    garch_results : ARCHModelResult
        Resultados do modelo GARCH
    horizon : int
        Horizonte de previsão em dias
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame com previsões
    """
    # Fazer previsão de variância
    forecasts = garch_results.forecast(horizon=horizon)
    
    # Extrair previsão de volatilidade (raiz quadrada da variância)
    forecast_vol = np.sqrt(forecasts.variance.iloc[-horizon:])
    
    # Criar datas futuras
    last_date = garch_results.data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
    
    # Criar DataFrame de previsão
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast_Volatility': forecast_vol.values
    })
    forecast_df.set_index('Date', inplace=True)
    
    return forecast_df

def plot_volatility_forecast(returns_df, garch_results, forecast_df):
    """
    Plota a previsão de volatilidade.
    
    Parâmetros:
    -----------
    returns_df : pd.DataFrame
        DataFrame com retornos históricos
    garch_results : ARCHModelResult
        Resultados do modelo GARCH
    forecast_df : pd.DataFrame
        DataFrame com previsões de volatilidade
    """
    # Obter volatilidade histórica
    historical_vol = garch_results.conditional_volatility
    
    # Criar figura
    plt.figure(figsize=(10, 6))
    
    # Plotar volatilidade histórica (últimos 90 dias)
    plt.plot(historical_vol.index[-90:], historical_vol[-90:], 
             color='blue', label='Volatilidade Histórica')
    
    # Plotar previsão
    plt.plot(forecast_df.index, forecast_df['Forecast_Volatility'], 
             color='red', label='Previsão de Volatilidade')
    
    # Adicionar intervalo de confiança (simulado)
    upper_bound = forecast_df['Forecast_Volatility'] * 1.2
    lower_bound = forecast_df['Forecast_Volatility'] * 0.8
    plt.fill_between(forecast_df.index, lower_bound, upper_bound, 
                     color='red', alpha=0.2, label='Intervalo de Confiança (80%)')
    
    # Adicionar linha vertical para separar histórico e previsão
    plt.axvline(x=historical_vol.index[-1], color='black', linestyle='--', alpha=0.5)
    
    plt.title('Previsão de Volatilidade para os Próximos 30 Dias', fontsize=14)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Volatilidade', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('volatility_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Função principal para executar a análise de volatilidade."""
    print("Iniciando modelagem de volatilidade com ARCH/GARCH...")
    
    # 1. Gerar dados simulados
    print("Gerando dados simulados...")
    returns_df = generate_simulated_returns(n_days=500)
    
    # 2. Visualizar retornos e volatilidade
    print("Plotando retornos e volatilidade...")
    plot_returns_and_volatility(returns_df)
    
    # 3. Ajustar modelo GARCH
    print("Ajustando modelo GARCH(1,1)...")
    garch_model, garch_results = fit_garch_model(returns_df['Returns'])
    
    # 4. Visualizar resultados do GARCH
    print("Visualizando resultados do modelo GARCH...")
    plot_garch_results(returns_df, garch_results)
    
    # 5. Prever volatilidade futura
    print("Prevendo volatilidade futura...")
    forecast_df = forecast_volatility(garch_results)
    plot_volatility_forecast(returns_df, garch_results, forecast_df)
    
    # 6. Salvar resultados em CSV
    returns_df.to_csv('simulated_returns.csv')
    forecast_df.to_csv('volatility_forecast.csv')
    
    # 7. Imprimir resumo do modelo GARCH
    print("\nResumo do modelo GARCH(1,1):")
    print(garch_results.summary().tables[0].as_text())
    print("\nParâmetros estimados:")
    print(garch_results.summary().tables[1].as_text())
    
    print("\nAnálise de volatilidade concluída com sucesso!")

if __name__ == "__main__":
    main()
