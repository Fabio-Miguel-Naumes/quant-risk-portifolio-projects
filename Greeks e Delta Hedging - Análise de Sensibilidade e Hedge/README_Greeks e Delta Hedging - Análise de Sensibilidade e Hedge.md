# Greeks e Delta Hedging - Análise de Sensibilidade e Hedge (Versão Simplificada)

![Greeks](https://img.shields.io/badge/Finanças-Greeks-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Quant](https://img.shields.io/badge/Área-Quant-orange)

## Visão Geral

Este projeto foca em dois conceitos importantes na gestão de risco de opções:

1.  **Greeks**: Medidas de sensibilidade do preço de uma opção a mudanças em diferentes fatores de mercado.
2.  **Delta Hedging**: Uma estratégia usada para reduzir ou eliminar o risco direcional de uma posição em opções, ajustando continuamente a posição no ativo subjacente.

Entender os Greeks e o Delta Hedging é essencial para:

- Gestão de risco de portfólios de derivativos
- Market making de opções
- Estratégias de trading sofisticadas

## Conceitos Teóricos Simplificados

### Greeks

Os "Greeks" quantificam como o preço de uma opção reage a pequenas mudanças nos parâmetros que o afetam. Os principais são:

-   **Delta (Δ)**: Mede a variação no preço da opção para uma variação de R$1 no preço do ativo subjacente. Varia de 0 a 1 para calls e de -1 a 0 para puts. Um delta de 0.6 significa que, se o preço do ativo subir R$1, o preço da call subirá aproximadamente R$0.60.

-   **Gamma (Γ)**: Mede a taxa de variação do Delta para uma variação de R$1 no preço do ativo subjacente. Indica quão rapidamente o Delta muda. É máximo quando a opção está "at-the-money" (preço do ativo próximo ao strike).

-   **Theta (Θ)**: Mede a variação no preço da opção devido à passagem do tempo (decaimento temporal). Geralmente é negativo para opções compradas, pois o valor temporal diminui à medida que o vencimento se aproxima.

-   **Vega (ν)**: Mede a variação no preço da opção para uma variação de 1% na volatilidade implícita do ativo subjacente. Opções têm valor positivo de Vega (preço sobe com aumento da volatilidade).

### Delta Hedging

Delta Hedging é uma estratégia dinâmica que visa manter o Delta total de um portfólio (que pode incluir opções e o ativo subjacente) próximo de zero. Isso torna o valor do portfólio insensível a pequenas mudanças no preço do ativo subjacente.

**Exemplo**: Se você vendeu uma opção de compra (call) com Delta de 0.6, você tem uma exposição negativa ao Delta (equivalente a estar vendido em 0.6 unidades do ativo). Para fazer o delta hedge, você compraria 0.6 unidades do ativo subjacente. O Delta total da posição combinada (opção vendida + ativo comprado) seria -0.6 + 0.6 = 0.

Como o Delta da opção muda com o preço do ativo e o tempo, a posição de hedge no ativo precisa ser ajustada (rebalanceada) periodicamente. O Gamma mede a frequência com que esses ajustes são necessários.

## Funcionalidades do Código Simplificado

Este projeto simplificado inclui:

1.  **Cálculo de Greeks**: Implementação das fórmulas para Delta, Gamma, Vega e Theta usando o modelo Black-Scholes.
2.  **Visualização de Greeks**: Gráficos mostrando como os Greeks variam com o preço do ativo subjacente.
3.  **Simulação de Preços**: Geração de um caminho simulado para o preço do ativo subjacente.
4.  **Simulação de Delta Hedging**: Demonstração passo a passo de uma estratégia de delta hedging para uma opção de compra vendida, incluindo rebalanceamento.
5.  **Visualização do Hedge**: Gráficos mostrando a evolução do preço do ativo, valor da opção, delta e valor do portfólio de hedge ao longo do tempo.

## Como Usar

```python
# Instalar dependências
pip install -r requirements.txt

# Executar o script
python greeks_and_delta_hedging_simplified.py
```

## Estrutura do Código

O código está organizado em funções bem documentadas:

- `black_scholes_call()`: Calcula o preço da call (reutilizada de outros projetos).
- `calculate_greeks()`: Calcula Delta, Gamma, Vega e Theta.
- `plot_all_greeks_vs_spot()`: Visualiza os greeks em função do preço do ativo.
- `generate_asset_price_path()`: Simula o caminho do preço do ativo.
- `simulate_delta_hedging()`: Simula a estratégia de delta hedging.
- `plot_delta_hedging_simulation()`: Visualiza os resultados da simulação de hedge.

## Visualizações Geradas

O script gera duas visualizações principais:

1.  **Greeks vs. Preço do Ativo**: Mostra como Delta, Gamma, Theta e Vega variam com o preço do ativo subjacente para opções de compra e venda.
2.  **Simulação de Delta Hedging**: Mostra a evolução do preço do ativo, valor da opção, delta e o valor do portfólio de hedge ao longo do tempo, ilustrando como o hedge tenta replicar o valor da opção.

## Requisitos

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Aplicações Práticas

### Market Makers
- Gerenciar o risco de inventário de opções mantendo posições delta-neutras.
- Capturar o spread entre compra e venda lucrando com o decaimento temporal (Theta) e Vega.

### Mesas de Derivativos em Bancos
- Fazer hedge de exposições complexas de produtos estruturados.
- Gerenciar o risco de portfólios de opções.

### Traders
- Implementar estratégias delta-neutras (ex: straddles, strangles) para lucrar com movimentos de volatilidade (Vega) ou decaimento temporal (Theta).
- Gerenciar o risco direcional de posições em opções.

## Extensões Possíveis

- Implementação de Gamma Hedging (neutralizar também o Gamma).
- Análise do impacto dos custos de transação no delta hedging.
- Comparação de diferentes frequências de rebalanceamento.
- Hedge de outros Greeks (Vega Hedging, Theta Hedging).

## Referências

1.  Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. 10th Edition. Pearson. (Capítulos sobre Greeks e Hedging)
2.  Taleb, N. N. (1997). *Dynamic Hedging: Managing Vanilla and Exotic Options*. Wiley.

---

*Este projeto foi desenvolvido como parte de um portfólio para demonstrar habilidades em finanças quantitativas, gestão de risco de derivativos e estratégias de hedge.*
