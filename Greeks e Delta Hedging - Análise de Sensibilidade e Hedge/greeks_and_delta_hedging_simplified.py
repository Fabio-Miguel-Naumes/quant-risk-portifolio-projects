"""
Greeks e Delta Hedging - Análise de Sensibilidade de Opções e Estratégias de Hedge (Versão Simplificada)
===================================================================================================

Este script implementa o cálculo dos "Greeks" (Delta, Gamma, Vega, Theta) para opções
e demonstra uma estratégia simples de Delta Hedging para neutralizar o risco direcional.

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

def black_scholes_call(S, K, T, r, sigma):
    """
    Calcula o preço de uma opção de compra (call) europeia usando o modelo Black-Scholes.
    
    Parâmetros:
    -----------
    S : float
        Preço atual do ativo subjacente
    K : float
        Preço de exercício (strike price)
    T : float
        Tempo até o vencimento em anos
    r : float
        Taxa de juros livre de risco (anual)
    sigma : float
        Volatilidade do ativo subjacente (anual)
        
    Retorna:
    --------
    float
        Preço teórico da opção de compra
    """
    # Verificar entradas
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    
    # Calcular d1 e d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calcular preço da call
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calcula os principais greeks para opções europeias.
    
    Parâmetros:
    -----------
    S : float
        Preço atual do ativo subjacente
    K : float
        Preço de exercício (strike price)
    T : float
        Tempo até o vencimento em anos
    r : float
        Taxa de juros livre de risco (anual)
    sigma : float
        Volatilidade do ativo subjacente (anual)
    option_type : str
        Tipo de opção ("call" ou "put")
        
    Retorna:
    --------
    dict
        Dicionário com os valores dos greeks
    """
    # Verificar entradas
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return {
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0
            }
        else:  # put
            return {
                "delta": -1.0 if S < K else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0
            }
    
    # Calcular d1 e d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calcular os greeks
    if option_type == "call":
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # Dividido por 100 para escala percentual
    
    if option_type == "call":
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    theta = theta / 365  # Convertido para base diária
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta
    }

def plot_all_greeks_vs_spot(S_range=np.linspace(70, 130, 100), K=100, T=1.0, r=0.05, sigma=0.2):
    """
    Plota todos os greeks em função do preço do ativo subjacente.
    
    Parâmetros:
    -----------
    S_range : array
        Faixa de preços do ativo subjacente
    K : float
        Preço de exercício
    T : float
        Tempo até o vencimento em anos
    r : float
        Taxa de juros livre de risco
    sigma : float
        Volatilidade do ativo
    """
    # Calcular greeks para calls e puts para cada valor de S
    call_greeks = [calculate_greeks(S, K, T, r, sigma, "call") for S in S_range]
    put_greeks = [calculate_greeks(S, K, T, r, sigma, "put") for S in S_range]
    
    # Extrair valores dos greeks
    call_delta = [g["delta"] for g in call_greeks]
    put_delta = [g["delta"] for g in put_greeks]
    gamma = [g["gamma"] for g in call_greeks]  # Gamma é igual para call e put
    call_theta = [g["theta"] for g in call_greeks]
    put_theta = [g["theta"] for g in put_greeks]
    vega = [g["vega"] for g in call_greeks]  # Vega é igual para call e put
    
    # Criar figura com múltiplos subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Delta
    axs[0, 0].plot(S_range, call_delta, "b-", linewidth=2, label="Delta Call")
    axs[0, 0].plot(S_range, put_delta, "r-", linewidth=2, label="Delta Put")
    axs[0, 0].set_title("Delta - Sensibilidade ao Preço do Ativo", fontsize=14)
    axs[0, 0].set_xlabel("Preço do Ativo (S)", fontsize=12)
    axs[0, 0].set_ylabel("Delta", fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].axvline(x=K, color="gray", linestyle="--", alpha=0.5)
    axs[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axs[0, 0].legend()
    
    # Gamma
    axs[0, 1].plot(S_range, gamma, "g-", linewidth=2)
    axs[0, 1].set_title("Gamma - Taxa de Variação do Delta", fontsize=14)
    axs[0, 1].set_xlabel("Preço do Ativo (S)", fontsize=12)
    axs[0, 1].set_ylabel("Gamma", fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].axvline(x=K, color="gray", linestyle="--", alpha=0.5)
    
    # Theta
    axs[1, 0].plot(S_range, call_theta, "b-", linewidth=2, label="Theta Call")
    axs[1, 0].plot(S_range, put_theta, "r-", linewidth=2, label="Theta Put")
    axs[1, 0].set_title("Theta - Sensibilidade ao Tempo", fontsize=14)
    axs[1, 0].set_xlabel("Preço do Ativo (S)", fontsize=12)
    axs[1, 0].set_ylabel("Theta (por dia)", fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].axvline(x=K, color="gray", linestyle="--", alpha=0.5)
    axs[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axs[1, 0].legend()
    
    # Vega
    axs[1, 1].plot(S_range, vega, "c-", linewidth=2)
    axs[1, 1].set_title("Vega - Sensibilidade à Volatilidade", fontsize=14)
    axs[1, 1].set_xlabel("Preço do Ativo (S)", fontsize=12)
    axs[1, 1].set_ylabel("Vega (por 1% de mudança em σ)", fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].axvline(x=K, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("all_greeks_vs_spot.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_asset_price_path(S0=100, mu=0.05, sigma=0.2, T=1.0, dt=1/252, seed=None):
    """
    Gera um caminho de preços para o ativo subjacente usando um processo de Wiener.
    
    Parâmetros:
    -----------
    S0 : float
        Preço inicial do ativo
    mu : float
        Retorno esperado anual
    sigma : float
        Volatilidade anual
    T : float
        Horizonte de tempo em anos
    dt : float
        Incremento de tempo (1/252 para dias úteis)
    seed : int
        Semente para geração de números aleatórios
        
    Retorna:
    --------
    tuple
        (array de tempos, array de preços)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Número de passos
    n_steps = int(T / dt)
    
    # Inicializar arrays
    times = np.linspace(0, T, n_steps + 1)
    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    
    # Gerar incrementos do processo de Wiener (movimento aleatório)
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    # Simular preços usando a fórmula do Movimento Browniano Geométrico
    for t in range(1, n_steps + 1):
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
    
    return times, prices

def simulate_delta_hedging(S0=100, K=100, T=0.25, r=0.05, sigma=0.2, dt=1/252, seed=None):
    """
    Simula uma estratégia de delta hedging para uma opção de compra.
    
    Parâmetros:
    -----------
    S0 : float
        Preço inicial do ativo
    K : float
        Preço de exercício
    T : float
        Tempo até o vencimento em anos (ex: 0.25 para 3 meses)
    r : float
        Taxa de juros livre de risco
    sigma : float
        Volatilidade do ativo
    dt : float
        Incremento de tempo (1/252 para dias úteis)
    seed : int
        Semente para geração de números aleatórios
        
    Retorna:
    --------
    dict
        Resultados da simulação
    """
    # Gerar caminho de preços
    times, prices = generate_asset_price_path(S0, r, sigma, T, dt, seed)
    
    # Número de passos
    n_steps = len(times) - 1
    
    # Inicializar arrays para armazenar resultados
    option_prices = np.zeros(n_steps + 1)
    deltas = np.zeros(n_steps + 1)
    hedge_positions = np.zeros(n_steps + 1)  # Quantidade do ativo para hedge
    cash_positions = np.zeros(n_steps + 1)   # Posição em caixa
    portfolio_values = np.zeros(n_steps + 1) # Valor do portfólio de hedge
    
    # Calcular preço inicial da opção e delta
    option_prices[0] = black_scholes_call(prices[0], K, T, r, sigma)
    deltas[0] = calculate_greeks(prices[0], K, T, r, sigma, "call")["delta"]
    
    # Inicializar posição de hedge (comprar delta * S0 do ativo)
    hedge_positions[0] = deltas[0]
    # Posição em caixa = Valor da opção vendida - Custo do hedge inicial
    cash_positions[0] = option_prices[0] - deltas[0] * prices[0]
    # Valor inicial do portfólio = Caixa + Valor do ativo
    portfolio_values[0] = cash_positions[0] + hedge_positions[0] * prices[0]
    
    # Simular estratégia de delta hedging
    for t in range(1, n_steps + 1):
        # Tempo restante até o vencimento
        tau = T - times[t]
        
        # Calcular novo preço da opção e delta
        if tau <= 1e-6: # Próximo do vencimento
            option_prices[t] = max(0, prices[t] - K)
            deltas[t] = 1.0 if prices[t] > K else 0.0
        else:
            option_prices[t] = black_scholes_call(prices[t], K, tau, r, sigma)
            deltas[t] = calculate_greeks(prices[t], K, tau, r, sigma, "call")["delta"]
        
        # Atualizar posição em caixa (com juros)
        cash_positions[t] = cash_positions[t-1] * np.exp(r * dt)
        
        # Calcular valor do portfólio antes do rebalanceamento
        # Valor = Caixa atual + Quantidade anterior do ativo * Preço atual do ativo
        portfolio_before = cash_positions[t] + hedge_positions[t-1] * prices[t]
        
        # Rebalancear portfólio: ajustar a quantidade do ativo para o novo delta
        # Custo/Receita do rebalanceamento = (Novo Delta - Delta Anterior) * Preço Atual
        rebalance_cost = (deltas[t] - hedge_positions[t-1]) * prices[t]
        
        # Atualizar posição em caixa após rebalanceamento
        cash_positions[t] = cash_positions[t] - rebalance_cost
        
        # Atualizar posição de hedge
        hedge_positions[t] = deltas[t]
        
        # Calcular valor final do portfólio no passo t
        portfolio_values[t] = cash_positions[t] + hedge_positions[t] * prices[t]
    
    # Calcular P&L (Lucro/Prejuízo) da estratégia
    # P&L = Valor final do portfólio de hedge - Valor final da opção
    pnl = portfolio_values[-1] - option_prices[-1]
    
    return {
        "times": times,
        "prices": prices,
        "option_prices": option_prices,
        "deltas": deltas,
        "hedge_positions": hedge_positions,
        "cash_positions": cash_positions,
        "portfolio_values": portfolio_values,
        "pnl": pnl
    }

def plot_delta_hedging_simulation(results):
    """
    Plota os resultados de uma simulação de delta hedging.
    
    Parâmetros:
    -----------
    results : dict
        Resultados da simulação
    """
    times = results["times"]
    prices = results["prices"]
    option_prices = results["option_prices"]
    deltas = results["deltas"]
    hedge_positions = results["hedge_positions"]
    portfolio_values = results["portfolio_values"]
    pnl = results["pnl"]
    
    # Criar figura com múltiplos subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Preço do ativo e valor da opção
    axs[0].plot(times, prices, "b-", linewidth=2, label="Preço do Ativo")
    axs[0].plot(times, option_prices, "r-", linewidth=2, label="Preço da Opção")
    axs[0].set_title("Preço do Ativo e Valor da Opção", fontsize=14)
    axs[0].set_ylabel("Preço", fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Delta e posição de hedge
    axs[1].plot(times, deltas, "g-", linewidth=2, label="Delta")
    # axs[1].plot(times, hedge_positions, "c--", linewidth=2, label="Posição de Hedge") # Pode ser redundante com Delta
    axs[1].set_title("Delta da Opção", fontsize=14)
    axs[1].set_ylabel("Delta", fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    # Valor do portfólio de hedge vs Valor da Opção
    axs[2].plot(times, portfolio_values, "m-", linewidth=2, label="Valor do Portfólio de Hedge")
    axs[2].plot(times, option_prices, "r--", linewidth=2, label="Valor da Opção")
    axs[2].set_title(f"Valor do Portfólio vs. Valor da Opção (P&L Final: {pnl:.4f})", fontsize=14)
    axs[2].set_xlabel("Tempo (anos)", fontsize=12)
    axs[2].set_ylabel("Valor", fontsize=12)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig("delta_hedging_simulation.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    """Função principal para executar a análise de Greeks e Delta Hedging."""
    print("Iniciando análise de Greeks e Delta Hedging...")
    
    # Definir parâmetros padrão
    S, K = 100, 100  # Preço do ativo e strike
    T, r, sigma = 1.0, 0.05, 0.2  # Tempo até vencimento, taxa de juros, volatilidade
    
    # 1. Calcular greeks para uma opção de compra
    greeks_call = calculate_greeks(S, K, T, r, sigma, "call")
    print("\nGreeks para uma Call (S=100, K=100, T=1, r=0.05, sigma=0.2):")
    for greek, value in greeks_call.items():
        print(f"{greek}: {value:.6f}")
        
    # 2. Plotar todos os greeks em função do preço do ativo
    print("\nPlotando greeks em função do preço do ativo...")
    plot_all_greeks_vs_spot()
    
    # 3. Simular delta hedging para uma opção de compra
    print("Simulando estratégia de delta hedging...")
    # Usar um tempo menor para a simulação (ex: 3 meses = 0.25 anos)
    hedging_results = simulate_delta_hedging(S0=100, K=100, T=0.25, r=0.05, sigma=0.2, seed=42)
    plot_delta_hedging_simulation(hedging_results)
    
    print(f"\nResultado da Simulação de Delta Hedging:")
    print(f"P&L Final (Erro de Hedge): {hedging_results["pnl"]:.6f}")
    
    print("\nAnálise de Greeks e Delta Hedging concluída com sucesso!")

if __name__ == "__main__":
    main()
