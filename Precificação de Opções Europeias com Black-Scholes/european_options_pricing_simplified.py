"""
Precificação de Opções Europeias com o Modelo Black-Scholes (Versão Simplificada)
==============================================================================

Este script implementa o modelo Black-Scholes para precificação de opções europeias,
um dos modelos fundamentais em finanças quantitativas. O modelo permite calcular
o preço teórico de opções de compra (call) e venda (put) europeias.

Versão simplificada para nível básico a intermediário.
"""

# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
warnings.filterwarnings('ignore')

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
    # N(d1) e N(d2) são funções de distribuição cumulativa normal
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calcula o preço de uma opção de venda (put) europeia usando o modelo Black-Scholes.
    
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
        Preço teórico da opção de venda
    """
    # Verificar entradas
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    
    # Calcular d1 e d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calcular preço da put
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price

def calculate_greeks(S, K, T, r, sigma):
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
        
    Retorna:
    --------
    dict
        Dicionário com os valores dos greeks
    """
    # Verificar entradas
    if T <= 0 or sigma <= 0:
        return {
            'delta_call': 1.0 if S > K else 0.0,
            'delta_put': -1.0 if S < K else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta_call': 0.0,
            'theta_put': 0.0
        }
    
    # Calcular d1 e d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calcular os greeks
    # Delta - sensibilidade ao preço do ativo
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    
    # Gamma - taxa de variação do delta (segunda derivada)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega - sensibilidade à volatilidade
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # Dividido por 100 para escala percentual
    
    # Theta - sensibilidade ao tempo
    theta_call = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_call = theta_call / 365  # Convertido para base diária
    
    theta_put = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta_put = theta_put / 365  # Convertido para base diária
    
    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'vega': vega,
        'theta_call': theta_call,
        'theta_put': theta_put
    }

def plot_option_prices_vs_spot(S_range=np.linspace(50, 150, 100), K=100, T=1.0, r=0.05, sigma=0.2):
    """
    Plota os preços de opções de compra e venda em função do preço do ativo subjacente.
    
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
    # Calcular preços de opções para cada valor de S
    call_prices = [black_scholes_call(S, K, T, r, sigma) for S in S_range]
    put_prices = [black_scholes_put(S, K, T, r, sigma) for S in S_range]
    
    # Calcular valores intrínsecos
    intrinsic_call = [max(0, S - K) for S in S_range]
    intrinsic_put = [max(0, K - S) for S in S_range]
    
    # Criar figura
    plt.figure(figsize=(10, 6))
    
    # Plotar preços de call
    plt.plot(S_range, call_prices, 'b-', linewidth=2, label='Preço Call (Black-Scholes)')
    plt.plot(S_range, intrinsic_call, 'b--', linewidth=1, label='Valor Intrínseco Call')
    
    # Plotar preços de put
    plt.plot(S_range, put_prices, 'r-', linewidth=2, label='Preço Put (Black-Scholes)')
    plt.plot(S_range, intrinsic_put, 'r--', linewidth=1, label='Valor Intrínseco Put')
    
    # Adicionar linha vertical no strike
    plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    plt.text(K+1, plt.ylim()[1]*0.9, f'K = {K}', fontsize=10)
    
    plt.title(f'Preços de Opções vs. Preço do Ativo (K={K}, T={T}, r={r}, σ={sigma})', fontsize=14)
    plt.xlabel('Preço do Ativo Subjacente (S)', fontsize=12)
    plt.ylabel('Preço da Opção', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('option_prices_vs_spot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_greeks(S_range=np.linspace(70, 130, 100), K=100, T=1.0, r=0.05, sigma=0.2):
    """
    Plota os principais greeks em função do preço do ativo subjacente.
    
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
    # Calcular greeks para cada valor de S
    greeks_list = [calculate_greeks(S, K, T, r, sigma) for S in S_range]
    
    # Extrair valores dos greeks
    delta_call = [g['delta_call'] for g in greeks_list]
    delta_put = [g['delta_put'] for g in greeks_list]
    gamma = [g['gamma'] for g in greeks_list]
    vega = [g['vega'] for g in greeks_list]
    theta_call = [g['theta_call'] for g in greeks_list]
    theta_put = [g['theta_put'] for g in greeks_list]
    
    # Criar figura com múltiplos subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Delta
    axs[0, 0].plot(S_range, delta_call, 'b-', linewidth=2, label='Delta Call')
    axs[0, 0].plot(S_range, delta_put, 'r-', linewidth=2, label='Delta Put')
    axs[0, 0].set_title('Delta - Sensibilidade ao Preço do Ativo', fontsize=14)
    axs[0, 0].set_xlabel('Preço do Ativo (S)', fontsize=12)
    axs[0, 0].set_ylabel('Delta', fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    axs[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[0, 0].legend()
    
    # Gamma
    axs[0, 1].plot(S_range, gamma, 'g-', linewidth=2)
    axs[0, 1].set_title('Gamma - Taxa de Variação do Delta', fontsize=14)
    axs[0, 1].set_xlabel('Preço do Ativo (S)', fontsize=12)
    axs[0, 1].set_ylabel('Gamma', fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    
    # Vega
    axs[1, 0].plot(S_range, vega, 'c-', linewidth=2)
    axs[1, 0].set_title('Vega - Sensibilidade à Volatilidade', fontsize=14)
    axs[1, 0].set_xlabel('Preço do Ativo (S)', fontsize=12)
    axs[1, 0].set_ylabel('Vega (por 1% de mudança em σ)', fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    
    # Theta
    axs[1, 1].plot(S_range, theta_call, 'b-', linewidth=2, label='Theta Call')
    axs[1, 1].plot(S_range, theta_put, 'r-', linewidth=2, label='Theta Put')
    axs[1, 1].set_title('Theta - Sensibilidade ao Tempo', fontsize=14)
    axs[1, 1].set_xlabel('Preço do Ativo (S)', fontsize=12)
    axs[1, 1].set_ylabel('Theta (por dia)', fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    axs[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('option_greeks.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_put_call_parity(S_range=np.linspace(70, 130, 100), K=100, T=1.0, r=0.05, sigma=0.2):
    """
    Demonstra a paridade put-call para opções europeias.
    
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
    # Calcular preços de opções para cada valor de S
    call_prices = [black_scholes_call(S, K, T, r, sigma) for S in S_range]
    put_prices = [black_scholes_put(S, K, T, r, sigma) for S in S_range]
    
    # Calcular lado esquerdo da equação de paridade: C - P
    left_side = [c - p for c, p in zip(call_prices, put_prices)]
    
    # Calcular lado direito da equação de paridade: S - K*exp(-rT)
    right_side = [S - K * np.exp(-r * T) for S in S_range]
    
    # Criar figura
    plt.figure(figsize=(10, 6))
    
    # Plotar ambos os lados da equação de paridade
    plt.plot(S_range, left_side, 'b-', linewidth=2, label='C - P')
    plt.plot(S_range, right_side, 'r--', linewidth=2, label='S - K*exp(-rT)')
    
    plt.title('Paridade Put-Call: C - P = S - K*exp(-rT)', fontsize=14)
    plt.xlabel('Preço do Ativo Subjacente (S)', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adicionar linha vertical no strike
    plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    plt.text(K+1, plt.ylim()[1]*0.9, f'K = {K}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('put_call_parity.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Função principal para executar a análise de precificação de opções europeias."""
    print("Iniciando análise de precificação de opções europeias com o modelo Black-Scholes...")
    
    # Definir parâmetros padrão
    S, K = 100, 100  # Preço do ativo e strike
    T, r, sigma = 1.0, 0.05, 0.2  # Tempo até vencimento, taxa de juros, volatilidade
    
    # 1. Calcular preços de opções
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    
    print(f"\nParâmetros: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Preço da Call: {call_price:.4f}")
    print(f"Preço da Put: {put_price:.4f}")
    
    # 2. Calcular greeks
    greeks = calculate_greeks(S, K, T, r, sigma)
    
    print("\nGreeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.6f}")
    
    # 3. Plotar preços de opções em função do preço do ativo
    print("\nGerando gráfico de preços de opções vs. preço do ativo...")
    plot_option_prices_vs_spot()
    
    # 4. Plotar greeks
    print("Gerando gráficos dos greeks...")
    plot_greeks()
    
    # 5. Demonstrar paridade put-call
    print("Gerando gráfico de paridade put-call...")
    plot_put_call_parity()
    
    # 6. Criar tabela de preços para diferentes strikes e vencimentos
    print("\nTabela de Preços de Opções para Diferentes Strikes:")
    strikes = [80, 90, 100, 110, 120]
    
    print("\nPreços de Call:")
    print(f"{'Strike':<10} {'Preço':<10}")
    print("-" * 20)
    for k in strikes:
        price = black_scholes_call(S, k, T, r, sigma)
        print(f"{k:<10} {price:<10.4f}")
    
    print("\nPreços de Put:")
    print(f"{'Strike':<10} {'Preço':<10}")
    print("-" * 20)
    for k in strikes:
        price = black_scholes_put(S, k, T, r, sigma)
        print(f"{k:<10} {price:<10.4f}")
    
    print("\nAnálise de precificação de opções europeias concluída com sucesso!")

if __name__ == "__main__":
    main()
