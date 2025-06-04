<<<<<<< HEAD
Comparação de Nove Algoritmos Clássicos de Aprendizado de Máquina em Conjuntos de Dados de Referência
Este repositório contém os resultados de um estudo comparativo aprofundado sobre o desempenho de nove algoritmos clássicos de aprendizado de máquina em diversos conjuntos de dados de referência. O objetivo foi investigar como as características intrínsecas dos dados influenciam a performance dos modelos.

📄 Sobre o Estudo
A seleção de algoritmos de aprendizado de máquina é crucial em problemas reais, pois o desempenho pode variar significativamente. Este estudo compara nove algoritmos clássicos (Árvores de Decisão, Regressão Linear, Regressão Logística, Redes Neurais Artificiais (MLP), Perceptron, SVM, Classificadores Bayesianos (GaussianNB), KNN e Random Forest) em nove conjuntos de dados públicos amplamente utilizados na literatura (Iris, Wine, Breast Cancer Wisconsin, Digits, Diabetes, Heart Disease, Parkinson, Titanic e Bank Marketing).

A avaliação foi realizada utilizando a métrica de acurácia média e testes estatísticos como o de Friedman e Nemenyi para garantir a robustez das conclusões.

🚀 Metodologia Experimental
Todos os experimentos foram conduzidos utilizando a biblioteca scikit-learn (versão 1.4.2).

Algoritmos Avaliados
Os seguintes algoritmos foram empregados:

Árvores de Decisão (DecisionTreeClassifier)
Regressão Linear (LinearRegression)
Regressão Logística (LogisticRegression)
Redes Neurais Artificiais (MLPClassifier)
Perceptron (Perceptron)
SVM (SVC)
Classificadores Bayesianos (GaussianNB)
KNN (KNeighborsClassifier)
Random Forest (RandomForestClassifier)
Hiperparâmetros
Para a maioria dos algoritmos, foram utilizados os valores padrão definidos pela scikit-learn. Ajustes mínimos foram feitos para garantir a convergência e estabilidade em alguns casos (ex: max_iter para LogisticRegression, MLPClassifier e Perceptron). Não foi realizado um ajuste fino exaustivo de hiperparâmetros.

Conjuntos de Dados
Os dados foram obtidos do repositório UCI e da própria scikit-learn.

Procedimentos
Para garantir uma avaliação robusta, foram adotados os seguintes procedimentos:

Validação cruzada estratificada com 10 folds: Assegura a manutenção da proporção das classes.
Normalização de atributos: Aplicada sempre que cabível para evitar distorções.
Métrica de desempenho: Acurácia média.
Análise estatística: Teste de Friedman e teste post-hoc de Nemenyi para identificar diferenças significativas.
Ferramentas
As bibliotecas utilizadas foram:

scikit-learn
pandas
seaborn
matplotlib
📊 Resultados e Discussão
Os resultados demonstram que conjuntos de dados com menor complexidade (como Iris e Wine) apresentam desempenho superior e mais consistente entre os algoritmos. Conjuntos mais complexos (como Digits, Titanic e Bank Marketing) exibem maior variabilidade de desempenho, ressaltando a dependência da escolha do algoritmo ideal em relação à complexidade dos dados, recursos computacionais e requisitos de precisão da aplicação.

Uma tabela detalhada comparando o desempenho de cada algoritmo em cada dataset, incluindo métricas como Acurácia, Precision (Weighted Avg), Recall (Weighted Avg), F1-Score (Weighted Avg) e MSE (para regressão), pode ser encontrada nos resultados do estudo.

👨‍🔬 Autor
Saulo Pereira da Silva

FACOM – Faculdade de Computação - Universidade Federal de Mato Grosso do Sul (UFMS) CEP 79070-900 – Campo Grande – MS - Brasil

Email: pereira.saulo@ufms.br

🤝 Como Contribuir
Sinta-se à vontade para explorar o código-fonte, reproduzir os experimentos e contribuir com insights ou melhorias. Se você tiver alguma sugestão ou encontrar algum problema, abra uma issue ou envie um pull request.
=======
# comparacao_algoritmos
>>>>>>> 39f834e5f70a3f3a02de62ddc8398c709f4995da
