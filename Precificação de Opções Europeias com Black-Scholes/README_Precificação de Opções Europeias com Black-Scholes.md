# Precificação de Opções Europeias com Black-Scholes (Versão Simplificada)

![Options](https://img.shields.io/badge/Finanças-Opções-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Quant](https://img.shields.io/badge/Área-Quant-orange)

## Visão Geral

Este projeto implementa o modelo Black-Scholes para calcular o preço teórico de opções europeias de compra (call) e venda (put). O modelo Black-Scholes é uma das ferramentas mais fundamentais em finanças quantitativas, usada para:

- Precificação de derivativos
- Gestão de risco
- Estratégias de trading

## Conceitos Teóricos Simplificados

### Opções Financeiras

Uma opção é um contrato que dá ao seu comprador o direito, mas não a obrigação, de comprar (opção de compra - call) ou vender (opção de venda - put) um ativo subjacente (como uma ação) a um preço específico (preço de exercício ou strike) em ou antes de uma data futura (data de vencimento).

- **Opção Europeia**: Só pode ser exercida na data de vencimento.
- **Opção Americana**: Pode ser exercida a qualquer momento até a data de vencimento.

### Modelo Black-Scholes

O modelo Black-Scholes fornece uma fórmula matemática para calcular o preço justo de uma opção europeia. Ele se baseia em algumas premissas, como:

- O preço do ativo subjacente segue um movimento browniano geométrico (distribuição log-normal).
- A taxa de juros livre de risco e a volatilidade do ativo são constantes.
- Não há custos de transação ou impostos.
- O ativo não paga dividendos durante a vida da opção.

As fórmulas para call e put são:

```
Call = S·N(d₁) - K·e⁻ʳᵀ·N(d₂)
Put = K·e⁻ʳᵀ·N(-d₂) - S·N(-d₁)
```

Onde:
- S: Preço atual do ativo
- K: Preço de exercício (strike)
- T: Tempo até o vencimento (em anos)
- r: Taxa de juros livre de risco (anual)
- σ: Volatilidade do ativo (anual)
- N(·): Função de distribuição cumulativa normal padrão
- d₁ e d₂: Parâmetros intermediários que dependem de S, K, T, r, σ

### Greeks

Os "Greeks" medem a sensibilidade do preço da opção a mudanças nos parâmetros do modelo:

- **Delta (Δ)**: Sensibilidade ao preço do ativo (S)
- **Gamma (Γ)**: Taxa de variação do Delta
- **Vega (ν)**: Sensibilidade à volatilidade (σ)
- **Theta (Θ)**: Sensibilidade ao tempo (T)

## Funcionalidades do Código Simplificado

Este projeto simplificado inclui:

1. **Cálculo de Preços**: Funções `black_scholes_call` e `black_scholes_put` para calcular os preços.

2. **Cálculo de Greeks**: Função `calculate_greeks` para calcular Delta, Gamma, Vega e Theta.

3. **Visualização de Preços**: Gráfico mostrando como os preços de call e put variam com o preço do ativo.

4. **Visualização de Greeks**: Gráficos mostrando como os Greeks variam com o preço do ativo.

5. **Demonstração da Paridade Put-Call**: Gráfico que ilustra a relação fundamental entre os preços de call e put europeias.

## Como Usar

```python
# Instalar dependências
pip install -r requirements.txt

# Executar o script
python european_options_pricing_simplified.py
```

## Estrutura do Código

O código está organizado em funções bem documentadas:

- `black_scholes_call()`: Calcula o preço da call
- `black_scholes_put()`: Calcula o preço da put
- `calculate_greeks()`: Calcula os principais greeks
- `plot_option_prices_vs_spot()`: Plota preços de opções vs. preço do ativo
- `plot_greeks()`: Plota os greeks vs. preço do ativo
- `plot_put_call_parity()`: Demonstra a paridade put-call

## Visualizações Geradas

O script gera três visualizações principais:

1. **Preços de Opções vs. Preço do Ativo**: Mostra como os preços teóricos de call e put mudam à medida que o preço do ativo subjacente varia, comparando com o valor intrínseco.

2. **Greeks vs. Preço do Ativo**: Mostra como Delta, Gamma, Vega e Theta variam com o preço do ativo, ajudando a entender o risco da opção.

3. **Paridade Put-Call**: Demonstra graficamente a relação `C - P = S - K*exp(-rT)`, que deve valer para evitar arbitragem.

## Requisitos

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Aplicações Práticas

### Bancos de Investimento
- Precificação de produtos estruturados
- Market making de opções
- Gestão de risco de mercado

### Asset Managers
- Estratégias de hedge com opções
- Geração de alfa através de estratégias com opções

### Fintechs
- Plataformas de trading de opções para varejo
- Ferramentas de análise de risco para investidores

## Extensões Possíveis

- Implementação de modelos para opções americanas (ex: Binomial, Trinomial)
- Incorporação de dividendos
- Modelos com volatilidade estocástica (ex: Heston)
- Cálculo de volatilidade implícita

## Referências

1. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.

3. Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. 10th Edition. Pearson.

---

*Este projeto foi desenvolvido como parte de um portfólio para demonstrar habilidades em finanças quantitativas e precificação de derivativos.*
