# Value at Risk (VaR) - Cálculo do Potencial de Perda (Versão Simplificada)

![VaR](https://img.shields.io/badge/Finanças-VaR-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Quant](https://img.shields.io/badge/Área-Quant-orange)

## Visão Geral

Este projeto implementa diferentes métodos para calcular o Value at Risk (VaR), uma medida estatística amplamente utilizada para quantificar o risco financeiro de um portfólio. O VaR responde à pergunta: "Qual é a perda máxima esperada em um determinado período de tempo, com um certo nível de confiança?"

O VaR é crucial para:

- Gestão de risco de mercado
- Alocação de capital
- Requisitos regulatórios (Basileia)
- Avaliação de desempenho ajustado ao risco

## Conceitos Teóricos Simplificados

### O que é VaR?

O Value at Risk (VaR) estima a perda potencial máxima que um portfólio pode sofrer em um determinado horizonte de tempo (ex: 1 dia, 10 dias) e com um nível de confiança específico (ex: 95%, 99%).

Por exemplo, um VaR de 1 dia de R$ 1 milhão com 95% de confiança significa que há 95% de chance de que a perda do portfólio não exceda R$ 1 milhão no próximo dia. Ou, alternativamente, há 5% de chance de que a perda seja maior que R$ 1 milhão.

### Métodos de Cálculo de VaR

Existem várias maneiras de calcular o VaR:

1.  **VaR Paramétrico (Variância-Covariância)**: Assume que os retornos do portfólio seguem uma distribuição normal. Calcula o VaR usando a média e o desvio padrão dos retornos históricos.
    - **Vantagem**: Simples e rápido de calcular.
    - **Desvantagem**: Pode subestimar o risco se os retornos não forem normais (ex: caudas pesadas).

2.  **VaR Histórico**: Usa a distribuição empírica dos retornos históricos. Ordena os retornos passados e encontra o percentil correspondente ao nível de confiança.
    - **Vantagem**: Não assume uma distribuição específica para os retornos.
    - **Desvantagem**: Assume que o passado é um bom preditor do futuro; sensível ao período histórico escolhido.

3.  **VaR por Simulação de Monte Carlo**: Simula milhares de cenários futuros possíveis para os retornos do portfólio com base em premissas estatísticas (média, desvio padrão, correlações). Calcula o VaR a partir da distribuição dos resultados simulados.
    - **Vantagem**: Flexível, pode modelar distribuições complexas e não-linearidades.
    - **Desvantagem**: Computacionalmente intensivo; depende da qualidade das premissas do modelo.

### Conditional Value at Risk (CVaR)

O CVaR, também conhecido como Expected Shortfall (ES), mede a perda esperada *dado que* a perda excedeu o nível do VaR. Ele fornece uma estimativa da magnitude das perdas na cauda da distribuição.

## Funcionalidades do Código Simplificado

Este projeto simplificado inclui:

1.  **Geração de Dados Simulados**: Criação de retornos para um portfólio de múltiplos ativos.
2.  **Cálculo de Retornos do Portfólio**: Combina os retornos dos ativos com base em pesos.
3.  **Visualização da Distribuição**: Histograma dos retornos do portfólio.
4.  **Cálculo de VaR**: Implementação dos métodos Paramétrico, Histórico e Monte Carlo.
5.  **Cálculo de CVaR**: Implementação do CVaR histórico.
6.  **Comparação de Métodos**: Gráfico comparando os resultados dos diferentes métodos de VaR.
7.  **Backtesting de VaR**: Avaliação do desempenho do modelo VaR histórico em prever perdas reais.

## Como Usar

```python
# Instalar dependências
pip install -r requirements.txt

# Executar o script
python value_at_risk_simplified.py
```

## Estrutura do Código

O código está organizado em funções bem documentadas:

- `generate_portfolio_returns()`: Gera retornos simulados para os ativos.
- `calculate_portfolio_returns()`: Calcula os retornos agregados do portfólio.
- `plot_returns_distribution()`: Visualiza a distribuição dos retornos.
- `calculate_parametric_var()`: Calcula o VaR paramétrico.
- `calculate_historical_var()`: Calcula o VaR histórico.
- `calculate_monte_carlo_var()`: Calcula o VaR por Monte Carlo.
- `calculate_conditional_var()`: Calcula o CVaR histórico.
- `plot_var_comparison()`: Compara os diferentes métodos de VaR.
- `plot_var_backtesting()`: Realiza e visualiza o backtesting do VaR.

## Visualizações Geradas

O script gera três visualizações principais:

1.  **Distribuição dos Retornos**: Histograma dos retornos do portfólio com uma curva normal ajustada.
2.  **Comparação de Métodos de VaR**: Gráfico de barras comparando os valores de VaR calculados pelos métodos paramétrico, histórico e Monte Carlo para diferentes níveis de confiança.
3.  **Backtesting de VaR**: Gráfico mostrando os retornos diários do portfólio em relação ao limite de VaR calculado (usando o método histórico) e destacando as violações (quando a perda real excedeu o VaR previsto).

## Requisitos

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Aplicações Práticas

### Bancos e Instituições Financeiras
- Cálculo de capital regulatório
- Limites de risco para mesas de trading
- Relatórios de risco para a alta administração

### Gestores de Ativos
- Avaliação do risco de diferentes estratégias de investimento
- Otimização de portfólios considerando o VaR
- Comunicação de risco aos investidores

### Empresas Não Financeiras
- Gestão de risco cambial e de commodities
- Avaliação de risco de projetos de investimento

## Extensões Possíveis

- Implementação de VaR baseado em modelos GARCH
- Cálculo de VaR para diferentes horizontes de tempo
- Análise de sensibilidade do VaR aos parâmetros
- Incorporação de VaR em otimização de portfólios

## Referências

1.  Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*. 3rd Edition. McGraw-Hill.
2.  Dowd, K. (2005). *Measuring Market Risk*. 2nd Edition. Wiley.
3.  McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*. Revised Edition. Princeton University Press.

---

*Este projeto foi desenvolvido como parte de um portfólio para demonstrar habilidades em finanças quantitativas e gestão de risco.*
