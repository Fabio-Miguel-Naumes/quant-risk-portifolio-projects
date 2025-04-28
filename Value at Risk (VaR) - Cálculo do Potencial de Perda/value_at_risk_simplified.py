"""
Value at Risk (VaR) - Cálculo do Potencial de Perda de um Portfólio (Versão Simplificada)
====================================================================================

Este script implementa diferentes métodos para calcular o Value at Risk (VaR),
uma medida estatística que quantifica o nível de risco financeiro de um portfólio
em um período de tempo específico.

Versão simplificada para nível básico a intermediário.
"""

# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
from datetime import datetime, timedelta

# Configurações de visualização
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
warnings.filterwarnings("ignore")

# Configurar seed para reprodutibilidade
np.random.seed(42)

def generate_portfolio_returns(n_days=500, n_assets=3):
    """
    Gera retornos diários simulados para um portfólio de ativos.
    
    Parâmetros:
    -----------
    n_days : int
        Número de dias a serem simulados
    n_assets : int
        Número de ativos no portfólio
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame com datas e retornos simulados para cada ativo
    """
    # Criar datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_days)
    
    # Definir parâmetros dos ativos
    mean_returns = np.random.uniform(0.0002, 0.0008, n_assets)
    volatilities = np.random.uniform(0.15, 0.35, n_assets) / np.sqrt(252)  # Convertido para base diária
    
    # Gerar matriz de correlação aleatória mas válida
    A = np.random.randn(n_assets, n_assets)
    correlation_matrix = np.corrcoef(A)
    
    # Calcular matriz de covariância
    volatility_matrix = np.diag(volatilities)
    covariance_matrix = volatility_matrix @ correlation_matrix @ volatility_matrix
    
    # Gerar retornos correlacionados
    returns = np.random.multivariate_normal(mean_returns, covariance_matrix, n_days)
    
    # Criar DataFrame
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    df = pd.DataFrame(returns, columns=asset_names)
    df["Date"] = dates
    df.set_index("Date", inplace=True)
    
    return df

def calculate_portfolio_returns(returns_df, weights=None):
    """
    Calcula os retornos do portfólio com base nos pesos dos ativos.
    
    Parâmetros:
    -----------
    returns_df : pd.DataFrame
        DataFrame com retornos dos ativos
    weights : array
        Pesos dos ativos no portfólio
        
    Retorna:
    --------
    pd.Series
        Série com retornos do portfólio
    """
    if weights is None:
        # Se não forem fornecidos pesos, usar pesos iguais
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
    
    # Calcular retornos do portfólio
    portfolio_returns = returns_df.dot(weights)
    
    return portfolio_returns

def plot_returns_distribution(returns, title="Distribuição dos Retornos"):
    """
    Plota a distribuição dos retornos com comparação com a distribuição normal.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    title : str
        Título do gráfico
    """
    plt.figure(figsize=(10, 6))
    
    # Histograma dos retornos
    sns.histplot(returns, kde=True, stat="density", alpha=0.6, color="blue")
    
    # Ajustar distribuição normal
    mu, sigma = norm.fit(returns)
    x = np.linspace(returns.min(), returns.max(), 1000)
    pdf_norm = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf_norm, "r-", linewidth=2, label=f"Normal: $\mu={mu:.4f}$, $\sigma={sigma:.4f}$")
    
    plt.title(title, fontsize=14)
    plt.xlabel("Retorno", fontsize=12)
    plt.ylabel("Densidade", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("returns_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

def calculate_parametric_var(returns, confidence_level=0.95):
    """
    Calcula o VaR paramétrico assumindo distribuição normal.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    confidence_level : float
        Nível de confiança (ex: 0.95 para 95%)
        
    Retorna:
    --------
    float
        Valor do VaR
    """
    # Calcular média e desvio padrão dos retornos
    mu = returns.mean()
    sigma = returns.std()
    
    # Calcular o z-score para o nível de confiança
    # norm.ppf é a função quantil (inversa da CDF)
    z = norm.ppf(1 - confidence_level)
    
    # Calcular VaR
    # VaR = -(Retorno Médio + Z-score * Desvio Padrão)
    var = -(mu + z * sigma)
    
    return var

def calculate_historical_var(returns, confidence_level=0.95):
    """
    Calcula o VaR histórico usando a distribuição empírica dos retornos.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    confidence_level : float
        Nível de confiança (ex: 0.95 para 95%)
        
    Retorna:
    --------
    float
        Valor do VaR
    """
    # Ordenar retornos do menor para o maior
    sorted_returns = np.sort(returns)
    
    # Calcular o índice correspondente ao percentil
    # Ex: Para 95% de confiança, queremos o 5º percentil (1 - 0.95 = 0.05)
    index = int(np.ceil((1 - confidence_level) * len(sorted_returns))) - 1
    index = max(0, index)  # Garantir que o índice não seja negativo
    
    # Obter o VaR (o retorno no índice calculado, com sinal invertido)
    var = -sorted_returns[index]
    
    return var

def calculate_monte_carlo_var(returns, confidence_level=0.95, n_simulations=10000):
    """
    Calcula o VaR usando simulação de Monte Carlo.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    confidence_level : float
        Nível de confiança (ex: 0.95 para 95%)
    n_simulations : int
        Número de simulações
        
    Retorna:
    --------
    float
        Valor do VaR
    """
    # Calcular média e desvio padrão dos retornos
    mu = returns.mean()
    sigma = returns.std()
    
    # Simular retornos futuros usando distribuição normal
    simulated_returns = np.random.normal(mu, sigma, n_simulations)
    
    # Calcular o VaR como o percentil da distribuição simulada
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    return var

def calculate_conditional_var(returns, confidence_level=0.95):
    """
    Calcula o Conditional Value at Risk (CVaR) ou Expected Shortfall (ES)
    usando o método histórico.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    confidence_level : float
        Nível de confiança (ex: 0.95 para 95%)
        
    Retorna:
    --------
    float
        Valor do CVaR
    """
    # Ordenar retornos
    sorted_returns = np.sort(returns)
    
    # Calcular o índice correspondente ao percentil do VaR
    index = int(np.ceil((1 - confidence_level) * len(sorted_returns))) - 1
    index = max(0, index)  # Garantir que o índice não seja negativo
    
    # Calcular a média dos retornos abaixo do VaR (na cauda esquerda)
    cvar = -np.mean(sorted_returns[:index+1])
    
    return cvar

def plot_var_comparison(returns, confidence_levels=[0.9, 0.95, 0.99]):
    """
    Plota uma comparação entre diferentes métodos de cálculo de VaR para vários níveis de confiança.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    confidence_levels : list
        Lista com níveis de confiança
    """
    # Calcular VaR para diferentes métodos e níveis de confiança
    results = []
    
    for cl in confidence_levels:
        parametric_var = calculate_parametric_var(returns, cl)
        historical_var = calculate_historical_var(returns, cl)
        monte_carlo_var = calculate_monte_carlo_var(returns, cl)
        
        results.append({
            "Confidence Level": f"{cl*100:.0f}%",
            "Parametric": parametric_var,
            "Historical": historical_var,
            "Monte Carlo": monte_carlo_var
        })
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    results_df.set_index("Confidence Level", inplace=True)
    
    # Plotar resultados
    plt.figure(figsize=(10, 6))
    
    ax = results_df.plot(kind="bar", width=0.8)
    
    plt.title("Comparação de Métodos de VaR por Nível de Confiança", fontsize=14)
    plt.xlabel("Nível de Confiança", fontsize=12)
    plt.ylabel("Value at Risk (VaR)", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(title="Método")
    
    # Adicionar valores nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8)
    
    plt.tight_layout()
    plt.savefig("var_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return results_df

def plot_var_backtesting(returns, window_size=250, confidence_level=0.95, method="historical"):
    """
    Realiza backtesting do VaR e plota os resultados.
    
    Parâmetros:
    -----------
    returns : pd.Series
        Série com retornos
    window_size : int
        Tamanho da janela móvel em dias
    confidence_level : float
        Nível de confiança (ex: 0.95 para 95%)
    method : str
        Método para calcular o VaR ("parametric", "historical", "monte_carlo")
        
    Retorna:
    --------
    dict
        Resultados do backtesting
    """
    if len(returns) <= window_size:
        raise ValueError("Série de retornos muito curta para o tamanho da janela especificado")
    
    # Inicializar arrays para armazenar resultados
    var_values = np.zeros(len(returns) - window_size)
    violations = np.zeros(len(returns) - window_size, dtype=bool)
    
    # Realizar backtesting
    for i in range(len(returns) - window_size):
        # Obter janela de retornos
        window_returns = returns.iloc[i:i+window_size]
        
        # Calcular VaR
        if method == "parametric":
            var = calculate_parametric_var(window_returns, confidence_level)
        elif method == "historical":
            var = calculate_historical_var(window_returns, confidence_level)
        elif method == "monte_carlo":
            var = calculate_monte_carlo_var(window_returns, confidence_level)
        else:
            raise ValueError("Método não reconhecido")
        
        # Armazenar VaR
        var_values[i] = var
        
        # Verificar violação (se o retorno real foi pior que o VaR previsto)
        actual_return = returns.iloc[i+window_size]
        violations[i] = actual_return < -var
    
    # Calcular estatísticas de violações
    expected_violations = (1 - confidence_level) * len(var_values)
    actual_violations = np.sum(violations)
    violation_ratio = actual_violations / expected_violations if expected_violations > 0 else np.nan
    
    # Criar DataFrame com resultados
    dates = returns.index[window_size:]
    results_df = pd.DataFrame({
        "Date": dates,
        "Return": returns.iloc[window_size:].values,
        "VaR": var_values,
        "Violation": violations
    })
    
    # Plotar resultados
    plt.figure(figsize=(12, 6))
    
    # Plotar retornos e VaR
    plt.plot(results_df["Date"], results_df["Return"], "b-", alpha=0.5, label="Retorno Diário")
    plt.plot(results_df["Date"], -results_df["VaR"], "r-", label=f"VaR ({confidence_level*100:.0f}%)")
    
    # Destacar violações
    violation_dates = results_df.loc[results_df["Violation"], "Date"]
    violation_returns = results_df.loc[results_df["Violation"], "Return"]
    plt.scatter(violation_dates, violation_returns, color="red", s=50, marker="o", label="Violação de VaR")
    
    plt.title(f"Backtesting de VaR - Método {method.capitalize()}\n"
              f"Violações Esperadas: {expected_violations:.1f}, Violações Reais: {actual_violations}, "
              f"Razão: {violation_ratio:.2f}", fontsize=14)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Retorno", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("var_backtesting.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        "results_df": results_df,
        "expected_violations": expected_violations,
        "actual_violations": actual_violations,
        "violation_ratio": violation_ratio
    }

def main():
    """Função principal para executar a análise de Value at Risk (VaR)."""
    print("Iniciando análise de Value at Risk (VaR)...")
    
    # 1. Gerar dados simulados
    print("Gerando retornos simulados para um portfólio...")
    returns_df = generate_portfolio_returns(n_days=500, n_assets=3)
    
    # 2. Calcular retornos do portfólio (pesos iguais)
    print("Calculando retornos do portfólio...")
    portfolio_returns = calculate_portfolio_returns(returns_df)
    
    # 3. Visualizar distribuição dos retornos
    print("Visualizando distribuição dos retornos...")
    plot_returns_distribution(portfolio_returns, "Distribuição dos Retornos do Portfólio")
    
    # 4. Calcular VaR usando diferentes métodos
    print("Calculando VaR usando diferentes métodos...")
    confidence_level = 0.95
    
    parametric_var = calculate_parametric_var(portfolio_returns, confidence_level)
    historical_var = calculate_historical_var(portfolio_returns, confidence_level)
    monte_carlo_var = calculate_monte_carlo_var(portfolio_returns, confidence_level)
    
    print(f"\nValue at Risk (VaR) com {confidence_level*100:.0f}% de confiança:")
    print(f"VaR Paramétrico: {parametric_var:.6f}")
    print(f"VaR Histórico: {historical_var:.6f}")
    print(f"VaR Monte Carlo: {monte_carlo_var:.6f}")
    
    # 5. Calcular CVaR (Expected Shortfall)
    print("\nCalculando Conditional Value at Risk (CVaR)...")
    cvar_historical = calculate_conditional_var(portfolio_returns, confidence_level)
    print(f"CVaR Histórico: {cvar_historical:.6f}")
    
    # 6. Comparar VaR para diferentes níveis de confiança
    print("\nComparando VaR para diferentes níveis de confiança...")
    var_comparison = plot_var_comparison(portfolio_returns)
    
    # 7. Realizar backtesting de VaR
    print("\nRealizando backtesting de VaR (método histórico)...")
    backtesting_results = plot_var_backtesting(portfolio_returns, window_size=250, 
                                              confidence_level=0.95, method="historical")
    
    # 8. Salvar resultados
    print("\nSalvando resultados...")
    returns_df.to_csv("portfolio_returns.csv")
    pd.Series(portfolio_returns).to_csv("portfolio_aggregated_returns.csv")
    var_comparison.to_csv("var_comparison.csv")
    
    print("\nAnálise de Value at Risk (VaR) concluída com sucesso!")

if __name__ == "__main__":
    main()
