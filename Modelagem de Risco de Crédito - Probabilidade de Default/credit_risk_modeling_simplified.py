"""
Modelagem de Risco de Crédito - Estimativa de Probabilidade de Default (Versão Simplificada)
===========================================================================================

Este script implementa um modelo de regressão logística para estimar a probabilidade
de default (PD) de clientes, um conceito fundamental em risco de crédito.

Versão simplificada para nível básico a intermediário.
"""

# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
import warnings

# Configurações de visualização
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
warnings.filterwarnings("ignore")

# Configurar seed para reprodutibilidade
np.random.seed(42)

def generate_synthetic_credit_data(n_samples=1000):
    """
    Gera dados sintéticos de crédito com características e status de default.
    
    Parâmetros:
    -----------
    n_samples : int
        Número de clientes (amostras) a serem gerados
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame com dados sintéticos de crédito
    """
    # Gerar características dos clientes
    # Renda anual (em milhares)
    income = np.random.lognormal(mean=4.0, sigma=0.5, size=n_samples) * 10
    # Dívida total (em milhares)
    debt = income * np.random.uniform(0.1, 0.8, size=n_samples) * np.random.choice([0.5, 1.5], size=n_samples, p=[0.8, 0.2])
    # Score de crédito (simplificado)
    credit_score = np.random.randint(500, 800, size=n_samples) - (debt / income) * 100
    credit_score = np.clip(credit_score, 300, 850) # Limitar score
    
    # Calcular probabilidade de default (simplificada)
    # Maior dívida/renda e menor score aumentam a probabilidade
    log_odds = -2.0 + 0.005 * debt - 0.01 * credit_score + 0.01 * (debt / (income + 1e-6)) * 100
    prob_default = 1 / (1 + np.exp(-log_odds))
    
    # Gerar status de default (0 = Não Default, 1 = Default)
    default_status = (np.random.rand(n_samples) < prob_default).astype(int)
    
    # Criar DataFrame
    df = pd.DataFrame({
        "Income": income,
        "Debt": debt,
        "CreditScore": credit_score,
        "DebtToIncomeRatio": debt / (income + 1e-6),
        "Default": default_status
    })
    
    return df

def plot_exploratory_analysis(df):
    """
    Realiza e plota uma análise exploratória básica dos dados.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com os dados de crédito
    """
    print("\nEstatísticas Descritivas:")
    print(df.describe())
    
    print(f"\nProporção de Default: {df["Default"].mean():.2%}")
    
    # Criar figura com subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histograma da Renda por Status de Default
    sns.histplot(data=df, x="Income", hue="Default", kde=True, ax=axs[0])
    axs[0].set_title("Distribuição da Renda por Default", fontsize=14)
    
    # Histograma do Score de Crédito por Status de Default
    sns.histplot(data=df, x="CreditScore", hue="Default", kde=True, ax=axs[1])
    axs[1].set_title("Distribuição do Score por Default", fontsize=14)
    
    # Scatter plot: Dívida vs Renda por Status de Default
    sns.scatterplot(data=df, x="Income", y="Debt", hue="Default", alpha=0.6, ax=axs[2])
    axs[2].set_title("Dívida vs Renda por Default", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("exploratory_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

def train_logistic_regression(df):
    """
    Treina um modelo de regressão logística para prever default.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com os dados de crédito
        
    Retorna:
    --------
    tuple
        (modelo treinado, X_test, y_test)
    """
    # Definir variáveis independentes (X) e dependente (y)
    features = ["Income", "Debt", "CreditScore", "DebtToIncomeRatio"]
    X = df[features]
    y = df["Default"]
    
    # Dividir dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTamanho do conjunto de treino: {X_train.shape[0]}")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")
    
    # Criar e treinar o modelo de Regressão Logística
    # solver=\'liblinear\' é bom para datasets menores
    # class_weight=\'balanced\' ajuda a lidar com desbalanceamento de classes (default é raro)
    model = LogisticRegression(solver=\'liblinear\', class_weight=\'balanced\', random_state=42)
    model.fit(X_train, y_train)
    
    print("\nModelo de Regressão Logística treinado.")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Avalia o desempenho do modelo treinado no conjunto de teste.
    
    Parâmetros:
    -----------
    model : LogisticRegression
        Modelo treinado
    X_test : pd.DataFrame
        Variáveis independentes do conjunto de teste
    y_test : pd.Series
        Variável dependente do conjunto de teste
        
    Retorna:
    --------
    dict
        Dicionário com métricas de avaliação
    """
    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Calcular probabilidades previstas (Probabilidade de Default - PD)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilidade da classe 1 (Default)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\nAvaliação do Modelo no Conjunto de Teste:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Plotar curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"Curva ROC (AUC = {auc_roc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Aleatório")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    plt.title("Curva ROC - Característica de Operação do Receptor", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "predicted_probabilities": y_prob
    }

def display_coefficients(model, features):
    """
    Exibe os coeficientes do modelo de regressão logística.
    
    Parâmetros:
    -----------
    model : LogisticRegression
        Modelo treinado
    features : list
        Lista de nomes das variáveis independentes
    """
    # Obter coeficientes e intercepto
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    print("\nCoeficientes do Modelo Logístico:")
    print(f"Intercepto: {intercept:.4f}")
    for feature, coef in zip(features, coefficients):
        print(f"{feature}: {coef:.4f}")
        
    # Interpretação (simplificada): Coeficientes positivos aumentam a log-odds (probabilidade) de default,
    # coeficientes negativos diminuem.

def main():
    """Função principal para executar a análise de risco de crédito."""
    print("Iniciando modelagem de risco de crédito (versão simplificada)...")
    
    # 1. Gerar dados sintéticos
    print("Gerando dados sintéticos de crédito...")
    credit_data = generate_synthetic_credit_data(n_samples=2000)
    
    # 2. Análise Exploratória
    print("Realizando análise exploratória...")
    plot_exploratory_analysis(credit_data)
    
    # 3. Treinar modelo de Regressão Logística
    print("\nTreinando modelo de Regressão Logística...")
    model, X_test, y_test = train_logistic_regression(credit_data)
    
    # 4. Avaliar modelo
    print("\nAvaliando o modelo treinado...")
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # 5. Exibir coeficientes
    print("\nExibindo coeficientes do modelo...")
    features = ["Income", "Debt", "CreditScore", "DebtToIncomeRatio"]
    display_coefficients(model, features)
    
    # 6. Salvar dados e resultados
    print("\nSalvando dados e probabilidades previstas...")
    credit_data.to_csv("synthetic_credit_data.csv", index=False)
    
    # Adicionar probabilidades previstas ao conjunto de teste para análise
    test_results = X_test.copy()
    test_results["Actual_Default"] = y_test
    test_results["Predicted_Probability_Default"] = evaluation_results["predicted_probabilities"]
    test_results.to_csv("test_predictions.csv", index=False)
    
    print("\nModelagem de risco de crédito (simplificada) concluída com sucesso!")

if __name__ == "__main__":
    main()
