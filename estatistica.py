import pandas as pd # Para manipulação de dados em DataFrames
from sklearn.model_selection import train_test_split # Para dividir dados em conjuntos de treino e teste
from sklearn.preprocessing import StandardScaler, LabelEncoder # Para normalização e codificação de variáveis categóricas
from sklearn.metrics import ( # Métricas de avaliação de modelos
    accuracy_score,          # Acurácia para classificação
    classification_report,   # Relatório detalhado para classificação (precision, recall, f1-score)
    mean_squared_error,      # Erro Quadrático Médio (MSE) para regressão
    mean_absolute_error,     # Erro Absoluto Médio (MAE) para regressão
    r2_score                 # Coeficiente de Determinação (R²) para regressão
)
from sklearn.datasets import ( # Datasets embutidos do scikit-learn
    load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
)

# --- Algoritmos de Classificação ---
# Importação dos modelos de classificação a serem testados
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Algoritmos de Regressão ---
# Importação dos modelos de regressão a serem testados
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt # Para criação de gráficos estáticos
import seaborn as sns # Para criação de gráficos estatísticos mais bonitos
import numpy as np # Para operações numéricas, como raiz quadrada
import os # Para interagir com o sistema operacional, como criar diretórios

# --- Funções Auxiliares para Carregamento e Pré-processamento de Datasets ---

def load_and_preprocess_iris():
    """
    Carrega e pré-processa o dataset Iris.
    Retorna X (features), y (target) e o tipo da tarefa ('classification').
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, 'classification'

def load_and_preprocess_wine():
    """
    Carrega e pré-processa o dataset Wine.
    Retorna X (features), y (target) e o tipo da tarefa ('classification').
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y, 'classification'

def load_and_preprocess_breast_cancer():
    """
    Carrega e pré-processa o dataset Breast Cancer.
    Retorna X (features), y (target) e o tipo da tarefa ('classification').
    """
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return X, y, 'classification'

def load_and_preprocess_digits():
    """
    Carrega e pré-processa o dataset Digits.
    Retorna X (features), y (target) e o tipo da tarefa ('classification').
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y, 'classification'

def load_and_preprocess_diabetes():
    """
    Carrega o dataset Diabetes do scikit-learn.
    Retorna X (features), y (target) e o tipo da tarefa ('regression').
    """
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X, y, 'regression'

def load_and_preprocess_heart_disease(file_path='heart_disease.csv'):
    """
    Carrega e pré-processa o dataset Heart Disease (UCI).
    Assumindo que o arquivo 'heart_disease.csv' (originalmente heart.dat) está no mesmo diretório.
    Download: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/
    """
    try:
        # Carrega o CSV, usando regex para espaços como separador e 'python' engine para robustez
        df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python', skipinitialspace=True)

        # Verifica se o DataFrame resultante é válido
        if df.empty or df.shape[1] < 2:
            raise ValueError(f"O arquivo '{file_path}' foi lido, mas resultou em um DataFrame com menos de 2 colunas. Verifique o formato do arquivo.")

        # A última coluna é o target. Mapeia valores (2 para 1 = doente, 1 para 0 = saudável)
        X = df.iloc[:, :-1].values # Todas as colunas exceto a última como features
        y = (df.iloc[:, -1] == 2).astype(int).values # Última coluna como target, convertendo 2 para 1 e 1 para 0
        return X, y, 'classification'
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado. Baixe-o do UCI (heart.dat) e renomeie para heart_disease.csv.")
        return None, None, None
    except Exception as e:
        print(f"Erro ao carregar ou pré-processar Heart Disease: {e}")
        return None, None, None

def load_and_preprocess_parkinson(file_path='parkinsons.csv'):
    """
    Carrega e pré-processa o dataset Parkinson (UCI).
    Assumindo que o arquivo 'parkinsons.csv' (originalmente parkinsons.data) está no mesmo diretório.
    Download: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/
    """
    try:
        # Carrega o arquivo CSV
        df = pd.read_csv(file_path)
        # 'name' é uma coluna de identificador e é descartada
        df = df.drop('name', axis=1)
        # 'status' é a variável target
        X = df.drop('status', axis=1).values
        y = df['status'].values
        return X, y, 'classification'
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado. Baixe-o do UCI (parkinsons.data) e renomeie para parkinsons.csv.")
        return None, None, None
    except Exception as e:
        print(f"Erro ao carregar ou pré-processar Parkinson: {e}")
        return None, None, None

def load_and_preprocess_titanic(file_path='titanic.csv'):
    """
    Carrega e pré-processa o dataset Titanic (Kaggle).
    Assumindo que o arquivo 'titanic.csv' (ou train.csv do Kaggle) está no mesmo diretório.
    Este é um pré-processamento simplificado para a tarefa de classificação.
    Download: https://www.kaggle.com/c/titanic/data (usar train.csv e renomear para titanic.csv)
    """
    try:
        df = pd.read_csv(file_path)
        # Seleciona colunas relevantes para a análise
        df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()

        # Trata valores ausentes: preenche 'Age' com a mediana e 'Embarked' com a moda
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        # Codifica variáveis categóricas: 'Sex' com LabelEncoder (binário)
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
        # 'Embarked' com One-Hot Encoding (pd.get_dummies) para múltiplas categorias
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) # drop_first evita multicolinearidade

        X = df.drop('Survived', axis=1).values # Todas as colunas exceto 'Survived' como features
        y = df['Survived'].values # 'Survived' como target
        return X, y, 'classification'
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado. Baixe-o do Kaggle (train.csv) e renomeie para titanic.csv.")
        return None, None, None
    except Exception as e:
        print(f"Erro ao carregar ou pré-processar Titanic: {e}")
        return None, None, None

def load_and_preprocess_bank_marketing(file_path='bank-full.csv'):
    """
    Carrega e pré-processa o dataset Bank Marketing (UCI).
    Assumindo que o arquivo 'bank-full.csv' está no mesmo diretório.
    Download: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-full.csv
    """
    try:
        df = pd.read_csv(file_path, sep=';') # Carrega CSV com separador ';'

        # Codifica a variável target 'y' ('yes' para 1, 'no' para 0)
        df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

        # Identifica colunas categóricas (tipo 'object')
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        # Remove a coluna 'y' da lista de categóricas, pois já foi tratada
        if 'y' in categorical_cols:
            categorical_cols.remove('y')

        # Aplica One-Hot Encoding nas variáveis categóricas restantes
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df_encoded.drop('y', axis=1).values # Features: todas as colunas exceto 'y'
        y = df_encoded['y'].values # Target: coluna 'y'
        return X, y, 'classification'
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado. Baixe-o do UCI.")
        return None, None, None
    except Exception as e:
        print(f"Erro ao carregar ou pré-processar Bank Marketing: {e}")
        return None, None, None

# --- Lista global para armazenar os resultados de todos os experimentos ---
all_results = []

# --- Função Principal para Treinamento e Avaliação de Modelos ---

def run_experiment_and_store_results(dataset_name, load_func, model_name, model_class, **model_params):
    """
    Executa um experimento de machine learning para um dado dataset e algoritmo.
    Realiza pré-processamento, split de dados, treinamento, previsão e avaliação.
    Armazena os resultados em uma lista global e gera gráficos específicos.

    Args:
        dataset_name (str): Nome do dataset (para impressão e salvamento de arquivos).
        load_func (function): Função para carregar e pré-processar o dataset.
        model_name (str): Nome do algoritmo (para impressão e salvamento de arquivos).
        model_class: Classe do modelo do scikit-learn (ex: DecisionTreeClassifier).
        **model_params: Parâmetros específicos para instanciar o modelo.
    """
    print(f"\n--- Dataset: {dataset_name} | Algoritmo: {model_name} ---")

    # Carrega e pré-processa os dados usando a função auxiliar
    X, y, task_type = load_func()

    # Se houver erro no carregamento, encerra a execução para este experimento
    if X is None or y is None:
        return

    # Divide os dados em conjuntos de treino (70%) e teste (30%)
    # random_state garante a reprodutibilidade da divisão
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializa e aplica StandardScaler para normalizar as features
    # fit_transform nos dados de treino e transform nos dados de teste para evitar vazamento de dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instancia o modelo com os parâmetros fornecidos
    model = model_class(**model_params)

    # Treina o modelo com os dados de treino normalizados
    model.fit(X_train_scaled, y_train)
    # Realiza previsões no conjunto de teste normalizado
    y_pred = model.predict(X_test_scaled)

    # Cria uma entrada de dicionário para armazenar os resultados deste experimento
    result_entry = {
        'Dataset': dataset_name,
        'Algoritmo': model_name,
        'Tipo da Tarefa': task_type
    }

    # --- Avaliação para Tarefas de Classificação ---
    if task_type == 'classification':
        # Calcula a acurácia
        accuracy = accuracy_score(y_test, y_pred)
        # Gera o relatório de classificação detalhado como um dicionário
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Imprime as métricas no console
        print(f"Acurácia: {accuracy:.4f}")
        print("Relatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))

        # Adiciona as métricas ao dicionário de resultados
        result_entry['Acuracia'] = accuracy
        # Tenta pegar métricas da "weighted avg" (para classificação multiclasse)
        if 'weighted avg' in report_dict:
            result_entry['Precision (Weighted Avg)'] = report_dict['weighted avg']['precision']
            result_entry['Recall (Weighted Avg)'] = report_dict['weighted avg']['recall']
            result_entry['F1-Score (Weighted Avg)'] = report_dict['weighted avg']['f1-score']
        # Se não houver "weighted avg" (comum em classificação binária), pega as métricas da classe '1'
        else:
            if '1' in report_dict:
                result_entry['Precision'] = report_dict['1']['precision']
                result_entry['Recall'] = report_dict['1']['recall']
                result_entry['F1-Score'] = report_dict['1']['f1-score']

        # --- Geração de Gráfico de Matriz de Confusão para Digits (exemplo) ---
        if dataset_name == 'Digits' and model_name == 'SVC':
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.arange(10), yticklabels=np.arange(10)) # Rótulos para dígitos 0-9
            plt.title(f'Matriz de Confusão para {model_name} no {dataset_name}')
            plt.xlabel('Previsão')
            plt.ylabel('Real')
            plt.savefig(f'./graficos/{dataset_name}_{model_name}_ConfusionMatrix.png')
            plt.close() # Fecha a figura para liberar memória

    # --- Avaliação para Tarefas de Regressão ---
    elif task_type == 'regression':
        # Calcula as métricas de regressão
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # RMSE é a raiz quadrada do MSE
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Imprime as métricas no console
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")

        # Adiciona as métricas ao dicionário de resultados
        result_entry['MSE'] = mse
        result_entry['RMSE'] = rmse
        result_entry['MAE'] = mae
        result_entry['R2'] = r2

        # --- Geração de Gráfico de Dispersão para Regressão (exclusivo para Diabetes) ---
        if dataset_name == 'Diabetes':
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6) # Pontos de valores reais vs. previstos
            # Adiciona uma linha de referência onde y_test = y_pred (predições perfeitas)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.title(f'Valores Reais vs. Previstos ({model_name} no {dataset_name})')
            plt.xlabel('Valores Reais')
            plt.ylabel('Valores Previstos')
            plt.grid(True)
            plt.savefig(f'./graficos/{dataset_name}_{model_name}_ScatterPlot.png')
            plt.close() # Fecha a figura para liberar memória

    # Adiciona a entrada de resultado à lista global de todos os resultados
    all_results.append(result_entry)


# --- Seção de Execução Principal dos Experimentos ---

# Cria o diretório 'graficos' se ele não existir, para salvar as imagens
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# --- Execução para o Dataset Iris (Classificação) ---
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(10,), max_iter=1000)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'SVC', SVC)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Iris', load_and_preprocess_iris, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Wine (Classificação) ---
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(20,), max_iter=1000)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'SVC', SVC)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Wine', load_and_preprocess_wine, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Breast Cancer (Classificação) ---
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(50,), max_iter=1000)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'SVC', SVC)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Breast Cancer', load_and_preprocess_breast_cancer, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Digits (Classificação) ---
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(100,), max_iter=1000)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'SVC', SVC)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Digits', load_and_preprocess_digits, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Diabetes (Regressão) ---
# Apenas LinearRegression é aplicável aqui, pois os outros são classificadores.
run_experiment_and_store_results('Diabetes', load_and_preprocess_diabetes, 'LinearRegression', LinearRegression)

# --- Execução para o Dataset Heart Disease (Classificação) ---
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(20,), max_iter=1000)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'SVC', SVC)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Heart Disease', load_and_preprocess_heart_disease, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Parkinson (Classificação) ---
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(20,), max_iter=1000)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'SVC', SVC)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Parkinson', load_and_preprocess_parkinson, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Titanic (Classificação) ---
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(20,), max_iter=1000)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'SVC', SVC)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Titanic', load_and_preprocess_titanic, 'RandomForestClassifier', RandomForestClassifier)

# --- Execução para o Dataset Bank Marketing (Classificação) ---
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'DecisionTreeClassifier', DecisionTreeClassifier)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'LogisticRegression', LogisticRegression, max_iter=1000)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'MLPClassifier', MLPClassifier, hidden_layer_sizes=(50,), max_iter=1000)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'Perceptron', Perceptron, max_iter=1000, tol=1e-3)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'SVC', SVC)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'GaussianNB', GaussianNB)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'KNeighborsClassifier', KNeighborsClassifier)
run_experiment_and_store_results('Bank Marketing', load_and_preprocess_bank_marketing, 'RandomForestClassifier', RandomForestClassifier)


# --- Salvando Todos os Resultados em um Arquivo Excel ---
# Verifica se a lista de resultados não está vazia antes de tentar salvar
if all_results:
    # Converte a lista de dicionários (all_results) em um DataFrame do Pandas
    df_results = pd.DataFrame(all_results)
    excel_file_path = 'resultados_ml_experimentos.xlsx'
    # Salva o DataFrame no formato Excel. index=False evita que o índice do DataFrame seja gravado como uma coluna.
    df_results.to_excel(excel_file_path, index=False)
    print(f"\nTodos os resultados foram salvos em '{excel_file_path}'")

# --- Lógica para Gerar Gráficos Comparativos após todos os Experimentos ---
# Gera gráficos apenas se houver resultados
if all_results:
    # Cria um DataFrame a partir dos resultados globais
    df_results = pd.DataFrame(all_results)

    # --- Gráficos Comparativos para Tarefas de Classificação ---
    # Filtra os resultados para incluir apenas tarefas de classificação
    classification_results = df_results[df_results['Tipo da Tarefa'] == 'classification'].copy()

    # Itera sobre cada dataset de classificação único para gerar um gráfico por dataset
    for dataset in classification_results['Dataset'].unique():
        # Seleciona o subconjunto de dados para o dataset atual
        subset = classification_results[classification_results['Dataset'] == dataset]
        
        # Define qual métrica será plotada (preferência para Acurácia, depois F1-Score Ponderado)
        metric_to_plot = ''
        y_label = ''
        if 'Acuracia' in subset.columns:
            metric_to_plot = 'Acuracia'
            y_label = 'Acurácia'
        elif 'F1-Score (Weighted Avg)' in subset.columns:
            metric_to_plot = 'F1-Score (Weighted Avg)'
            y_label = 'F1-Score (Média Ponderada)'
        else:
            # Se nenhuma das métricas esperadas estiver presente, pula este dataset
            continue

        # Cria o gráfico de barras comparativo para o dataset atual
        plt.figure(figsize=(12, 6)) # Define o tamanho da figura
        sns.barplot(x='Algoritmo', y=metric_to_plot, data=subset) # Cria o gráfico de barras
        plt.title(f'{y_label} dos Algoritmos no Dataset {dataset}') # Título do gráfico
        plt.xlabel('Algoritmo') # Rótulo do eixo X
        plt.ylabel(y_label) # Rótulo do eixo Y
        plt.ylim(0.0, 1.05) # Define o limite do eixo Y de 0 a 1.05 para melhor visualização
        plt.xticks(rotation=45, ha='right') # Gira os rótulos do eixo X para melhor legibilidade
        plt.tight_layout() # Ajusta o layout para evitar sobreposição
        plt.savefig(f'./graficos/{dataset}_Comparativo_Classificacao.png') # Salva o gráfico como imagem
        plt.close() # Fecha a figura para liberar memória

    # --- Gráficos Comparativos para Tarefas de Regressão ---
    # Filtra os resultados para incluir apenas tarefas de regressão
    regression_results = df_results[df_results['Tipo da Tarefa'] == 'regression'].copy()
    
    # Gera o gráfico apenas se houver resultados de regressão
    if not regression_results.empty:
        plt.figure(figsize=(10, 6))
        # Plota RMSE como a métrica principal para o comparativo de regressão
        sns.barplot(x='Algoritmo', y='RMSE', data=regression_results)
        plt.title('RMSE dos Algoritmos no Dataset Diabetes (Regressão)')
        plt.xlabel('Algoritmo')
        plt.ylabel('Root Mean Squared Error (RMSE)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('./graficos/Diabetes_Comparativo_Regressao.png')
        plt.close() # Fecha a figura para liberar memória