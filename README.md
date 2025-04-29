# Portfólio de Projetos Quantitativos para Mercado Financeiro

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Quantitative-green)
![Risk](https://img.shields.io/badge/Risk-Management-red)
![Banking](https://img.shields.io/badge/Industry-Banking-purple)
![Investment](https://img.shields.io/badge/Focus-Investment-orange)

## Sobre este Repositório

Bem-vindo ao meu portfólio de projetos quantitativos para o mercado financeiro. Este repositório contém implementações em Python de modelos e técnicas fundamentais utilizadas por profissionais quantitativos em bancos de investimento, gestoras de ativos e fintechs.

Cada projeto foi desenvolvido com foco em clareza, documentação detalhada e visualizações de alta qualidade, demonstrando competências técnicas e conhecimento teórico relevantes para posições na área Quant.

## Projetos Incluídos

### 1. Modelagem de Volatilidade com ARCH/GARCH

![Volatility](https://img.shields.io/badge/Área-Volatilidade-blue)

Implementação de modelos ARCH (Autoregressive Conditional Heteroskedasticity) e GARCH (Generalized ARCH) para análise e previsão de volatilidade em séries temporais financeiras.

**Principais funcionalidades:**
- Geração de séries temporais com clusters de volatilidade
- Ajuste de modelos GARCH(1,1) e variantes
- Previsão de volatilidade futura com intervalos de confiança
- Visualizações detalhadas de resultados e diagnósticos

**Aplicações no mercado:** Gestão de risco, precificação de derivativos, otimização de portfólios, estratégias de trading baseadas em volatilidade.

[Ver projeto →](./volatility_modeling)

### 2. Precificação de Opções Europeias com Black-Scholes

![Options](https://img.shields.io/badge/Área-Derivativos-blue)

Implementação completa do modelo Black-Scholes para precificação de opções europeias, incluindo cálculo de greeks e análise de sensibilidade.

**Principais funcionalidades:**
- Cálculo preciso de preços de opções de compra e venda
- Visualizações 3D de superfícies de preços e volatilidade
- Análise de volatilidade implícita
- Demonstração da paridade put-call

**Aplicações no mercado:** Precificação de derivativos, estruturação de produtos, arbitragem, market making.

[Ver projeto →](./european_options_pricing)

### 3. Value at Risk (VaR) - Gestão de Risco de Mercado

![Risk](https://img.shields.io/badge/Área-Risco_de_Mercado-blue)

Implementação de diferentes metodologias para cálculo do Value at Risk (VaR), uma medida estatística fundamental para quantificação de risco financeiro.

**Principais funcionalidades:**
- Cálculo de VaR por métodos paramétrico, histórico e Monte Carlo
- Backtesting rigoroso com análise de violações
- Cálculo de Conditional Value at Risk (Expected Shortfall)
- Análise de cenários de stress

**Aplicações no mercado:** Gestão de risco de mercado, alocação de capital, compliance regulatório, relatórios para comitês de risco.

[Ver projeto →](./value_at_risk)

### 4. Greeks e Delta Hedging - Gestão de Risco de Derivativos

![Derivatives](https://img.shields.io/badge/Área-Hedge-blue)

Implementação do cálculo dos "Greeks" (Delta, Gamma, Vega, Theta, Rho) para opções e demonstração de estratégias de Delta Hedging para neutralizar o risco direcional.

**Principais funcionalidades:**
- Cálculo e visualização dos cinco principais Greeks
- Simulação dinâmica de estratégias de delta hedging
- Análise do impacto da frequência de rebalanceamento
- Implementação de estratégias comuns com opções

**Aplicações no mercado:** Gestão de risco de derivativos, market making, estruturação de produtos, trading proprietário.

[Ver projeto →](./greeks_and_delta_hedging)

### 5. Modelagem de Risco de Crédito - Probabilidade de Default

![Credit](https://img.shields.io/badge/Área-Risco_de_Crédito-blue)

Implementação de modelos para estimar a probabilidade de default (PD) de contrapartes, utilizando regressão logística e técnicas de machine learning.

**Principais funcionalidades:**
- Geração de dados sintéticos para modelagem de crédito
- Treinamento de modelos de regressão logística e comparativos
- Cálculo de perda esperada (Expected Loss)
- Desenvolvimento de scorecard de crédito

**Aplicações no mercado:** Concessão de crédito, precificação baseada em risco, cálculo de capital regulatório, gestão de portfólios de crédito.

[Ver projeto →](./credit_risk_modeling)

## Competências Demonstradas

### Programação e Análise de Dados
- **Python**: NumPy, Pandas, SciPy, Statsmodels
- **Visualização**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, modelos estatísticos

### Finanças Quantitativas
- Modelagem estocástica e processos aleatórios
- Precificação de derivativos e produtos estruturados
- Análise de séries temporais financeiras
- Gestão de risco de mercado e crédito
- Otimização de portfólios

### Matemática e Estatística
- Cálculo estocástico e equações diferenciais
- Métodos numéricos e simulação de Monte Carlo
- Inferência estatística e modelagem preditiva
- Regressão e classificação

## Instalação e Uso

Cada projeto possui seu próprio README com instruções detalhadas de instalação e uso. Em geral, você pode seguir estes passos:

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/quant-portfolio.git
cd quant-portfolio

# Entrar no diretório de um projeto específico
cd volatility_modeling

# Instalar dependências
pip install -r requirements.txt

# Executar o script principal
python volatility_modeling.py
```

## Estrutura do Repositório

```
quant-portfolio/
│
├── volatility_modeling/
│   ├── volatility_modeling.py
│   ├── README.md
│   └── requirements.txt
│
├── european_options_pricing/
│   ├── european_options_pricing.py
│   ├── README.md
│   └── requirements.txt
│
├── value_at_risk/
│   ├── value_at_risk.py
│   ├── README.md
│   └── requirements.txt
│
├── greeks_and_delta_hedging/
│   ├── greeks_and_delta_hedging.py
│   ├── README.md
│   └── requirements.txt
│
├── credit_risk_modeling/
│   ├── credit_risk_modeling.py
│   ├── README.md
│   └── requirements.txt
│
├── README.md
└── LICENSE
```

## Relevância para o Mercado Financeiro

Estes projetos demonstram competências técnicas e conhecimentos teóricos diretamente aplicáveis a diversas funções no mercado financeiro:

### Bancos de Investimento
- Precificação e estruturação de derivativos
- Gestão de risco de mercado e crédito
- Desenvolvimento de modelos quantitativos
- Análise de investimentos

### Asset Managers e Hedge Funds
- Desenvolvimento de estratégias quantitativas
- Otimização de portfólios
- Análise de risco e retorno
- Backtesting de estratégias

### Fintechs e Corretoras
- Desenvolvimento de algoritmos de trading
- Criação de ferramentas de análise para clientes
- Sistemas de gestão de risco
- Automação de processos financeiros

## Próximos Passos

Estou constantemente aprimorando e expandindo este portfólio. Projetos futuros incluirão:

- Otimização de portfólios com diferentes funções objetivo
- Estratégias de trading algorítmico e backtesting
- Modelos de machine learning para previsão de mercado
- Precificação de derivativos exóticos
- Análise de dados alternativos para decisões de investimento

## Contato

Estou aberto a oportunidades na área quantitativa em instituições financeiras. Sinta-se à vontade para entrar em contato:

- **LinkedIn**: [Seu Perfil LinkedIn](https://www.linkedin.com/in/seu-perfil/)
- **Email**: seu.email@exemplo.com

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
