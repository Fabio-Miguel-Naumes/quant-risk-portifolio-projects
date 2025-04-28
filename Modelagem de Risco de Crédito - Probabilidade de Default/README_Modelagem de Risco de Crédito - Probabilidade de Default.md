# Modelagem de Risco de Crédito - Probabilidade de Default (Versão Simplificada)

![Credit](https://img.shields.io/badge/Finanças-Crédito-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Quant](https://img.shields.io/badge/Área-Quant-orange)

## Visão Geral

Este projeto implementa um modelo de regressão logística para estimar a probabilidade de default (PD) de clientes, um conceito fundamental em risco de crédito. A modelagem de risco de crédito é essencial para:

- Decisões de concessão de crédito
- Precificação de empréstimos baseada em risco
- Cálculo de provisões e capital regulatório
- Gestão de portfólios de crédito

## Conceitos Teóricos Simplificados

### Risco de Crédito

O risco de crédito é a possibilidade de perda resultante da falha de um devedor em cumprir suas obrigações financeiras. Em modelagem de risco de crédito, geralmente trabalhamos com três componentes principais:

1. **Probabilidade de Default (PD)**: A probabilidade de um cliente não pagar suas obrigações dentro de um período específico (geralmente 12 meses).

2. **Exposição no Momento do Default (EAD)**: O valor que estaria em risco se ocorresse o default.

3. **Perda em Caso de Default (LGD)**: A proporção da exposição que seria efetivamente perdida se ocorresse o default.

A **Perda Esperada (Expected Loss - EL)** é calculada como:
```
EL = PD × EAD × LGD
```

### Regressão Logística para Modelagem de PD

A regressão logística é uma técnica estatística amplamente utilizada para estimar a probabilidade de default. Ela modela a relação entre um conjunto de características do cliente (variáveis independentes) e a probabilidade de default (variável dependente binária: 0 = não default, 1 = default).

A fórmula básica da regressão logística é:
```
log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Onde:
- p: probabilidade de default
- β₀: intercepto
- βᵢ: coeficientes
- Xᵢ: variáveis explicativas (características do cliente)

Resolvendo para p, temos:
```
p = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
```

## Funcionalidades do Código Simplificado

Este projeto simplificado inclui:

1. **Geração de Dados Sintéticos**: Criação de dados simulados de clientes com características financeiras e status de default.

2. **Análise Exploratória**: Visualizações para entender a distribuição das variáveis e suas relações com o default.

3. **Modelagem com Regressão Logística**: Treinamento de um modelo para prever a probabilidade de default.

4. **Avaliação do Modelo**: Métricas de desempenho como acurácia, AUC-ROC e matriz de confusão.

5. **Interpretação dos Coeficientes**: Análise do impacto de cada variável na probabilidade de default.

## Como Usar

```python
# Instalar dependências
pip install -r requirements.txt

# Executar o script
python credit_risk_modeling_simplified.py
```

## Estrutura do Código

O código está organizado em funções bem documentadas:

- `generate_synthetic_credit_data()`: Gera dados sintéticos de crédito.
- `plot_exploratory_analysis()`: Realiza análise exploratória básica.
- `train_logistic_regression()`: Treina o modelo de regressão logística.
- `evaluate_model()`: Avalia o desempenho do modelo.
- `display_coefficients()`: Exibe e interpreta os coeficientes do modelo.

## Visualizações Geradas

O script gera duas visualizações principais:

1. **Análise Exploratória**: Três gráficos mostrando a distribuição da renda, score de crédito e a relação entre dívida e renda, todos segmentados por status de default.

2. **Curva ROC**: Gráfico mostrando o desempenho do modelo em termos de sensibilidade vs. especificidade, com o valor da área sob a curva (AUC).

## Requisitos

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## Aplicações Práticas

### Bancos e Instituições Financeiras
- Automação de decisões de crédito
- Precificação baseada em risco
- Cálculo de provisões e capital regulatório

### Fintechs e Empresas de Crédito
- Desenvolvimento de modelos de scoring
- Otimização de estratégias de aquisição de clientes
- Personalização de ofertas de crédito

### Investidores
- Avaliação de risco de títulos de dívida
- Análise de risco de contraparte

## Extensões Possíveis

- Incorporação de técnicas de machine learning mais avançadas (Random Forest, Gradient Boosting)
- Desenvolvimento de scorecards de crédito
- Implementação de modelos para LGD e EAD
- Análise de sobrevivência para modelagem de tempo até o default

## Referências

1. Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*. 2nd Edition. Wiley.

2. Baesens, B., Roesch, D., & Scheule, H. (2016). *Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS*. Wiley.

3. Thomas, L. C., Crook, J. N., & Edelman, D. B. (2017). *Credit Scoring and Its Applications*. 2nd Edition. SIAM.

---

*Este projeto foi desenvolvido como parte de um portfólio para demonstrar habilidades em finanças quantitativas e análise de risco de crédito.*
