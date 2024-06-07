# Global-Solution-IA

## Relatório Técnico: Previsão da Abundância de Resíduos Plásticos

# Descrição do Problema
A proliferação de lixo plástico nos oceanos e praias representa uma grave ameaça ambiental, impactando negativamente a vida marinha e os ecossistemas costeiros. O objetivo deste desafio é desenvolver modelos de Machine Learning que possam prever a abundância de resíduos plásticos em diferentes locais com base em fatores geográficos, utilizando um dataset com dados de pedaços de micro plásticos por metro quadrado (KM2_pieces_plasticpolution), latitude e longitude.

# Metodologia Utilizada

# Exploração dos Dados:
Visualização dos primeiros e últimos registros.
Descrição estatística dos dados.
Tamanho do conjunto de dados.

# Visualização de Dados:
Matriz de correlação.
Gráfico de dispersão entre latitude e longitude.
Gráfico de dispersão entre as variáveis.

# Preparação dos Dados:
Separação dos dados de entrada (latitude e longitude) e de saída (quantidade de micro plásticos por metro quadrado).
Divisão dos dados em conjuntos de treinamento e teste.

# Modelagem:
Treinamento de um modelo KNeighborsClassifier (KNN) e de um modelo de Regressão Linear utilizando os métodos .fit e .predict.
Avaliação dos modelos com métricas de erro.

# Avaliação e Visualização dos Resultados:
Cálculo do Erro Quadrático Médio (MSE), Raiz do Erro Quadrático Médio (RMSE) e Coeficiente de Determinação (R²).
Criação de DataFrames para visualização das métricas.

# Código para Otimização da Velocidade de Execução:
Implementação de um pipeline otimizado para a execução dos modelos.




# Resultados Obtidos

Exploração dos Dados

# Primeiros Registros:
#visualizar os primeiros registros 
dados.head()

# Últimos Registros:
#visualizar os ultimos registros
dados.tail()

# Descrição dos Dados:
#estatisticas descritivas dos atributos
dados.describe()


# Visualização dos Dados
Matriz de Correlação:

corr_matrix = dados[['Pieces_KM2', 'Latitude', 'Longitude']].corr().round(4)
corr_matrix
#Gerando um heatmap com seaborn
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

# Gráfico de Dispersão (Latitude vs Longitude):
Bax = sns.pairplot(dados, y_vars='Pieces_KM2', x_vars=['Latitude', 'Longitude'])
Bax.fig.suptitle('Dispersion between variables', fontsize=14, y=1.05)
Bax


# Preparação dos Dados
# Separação dos Dados de Entrada e Saída:

# Separação dos dados de entrada (FEATURES) e dados de saída (TARGET)
 #X maiúsculo ----> features / variáveis independentes
X = dados[['Latitude', 'Longitude']]

#y minúsculo ----> target / variável dependente
y = dados[['Pieces_KM2']]
#visualizando os tres primeiros e tres ultimos dados de X
X.values
#visualizando os tres primeiros e tres ultimos dados de y
y.values

# Divisão dos Dados em Conjuntos de Treinamento e Teste:
#Separação de dados de treino e teste
#X_treino, y_treino ----> para treinar o modelo com método .fit
#X_teste ----> para gerar as previsões com método .predict
#y_teste ----> para avaliar as previsões

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Modelagem e Treinamento
Modelo KNeighborsClassifier:

knn_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

#Cálculo do MSE (Erro Quadrático Médio)
MSE_2 = mean_squared_error(y_test, y_predict_knn)
#Cálculo do RMSE (Raiz do Erro Quadrático Médio)
RMSE_2 = np.sqrt(mean_squared_error(y_test, y_predict_knn))
#Cálculo do R² (Coeficiente de Determinação)
R2_2 = r2_score(y_test, y_predict_knn)

#Criação de um DataFrame com as métricas
pd.DataFrame([MSE_2, RMSE_2, R2_2], ['MSE', 'RMSE', 'R²'], columns=['KNN'])



# Otimização da Velocidade de Execução
Para otimizar a velocidade de execução dos modelos, implementamos um pipeline que divide o conjunto de dados em treinamento e teste, treina dois modelos diferentes com os dados de treinamento e gera previsões nos dados de teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

y_predict_knn = knn_model.predict(X_test)
y_predict_lr = lr_model.predict(X_test)
MSE_2 = mean_squared_error(y_test, y_predict_knn)
RMSE_2 = np.sqrt(mean_squared_error(y_test, y_predict_knn))
R2_2 = r2_score(y_test, y_predict_knn)

pd.DataFrame([MSE_2, RMSE_2, R2_2], ['MSE', 'RMSE', 'R²'], columns=['KNN'])

MSE_2 = mean_squared_error(y_test, y_predict_lr)
RMSE_2 = np.sqrt(mean_squared_error(y_test, y_predict_lr))
R2_2 = r2_score(y_test, y_predict_lr)

pd.DataFrame([MSE_2, RMSE_2, R2_2], ['MSE', 'RMSE', 'R²'], columns=['LINEAR REGRESSION'])

# Conclusões
A partir dos resultados obtidos, podemos observar o desempenho de dois modelos diferentes na previsão da abundância de resíduos plásticos com base em dados geográficos. O modelo de Regressão Linear e o KNeighborsClassifier apresentaram métricas diferentes de desempenho, oferecendo uma visão clara sobre qual modelo se adapta melhor ao problema em questão. A otimização do pipeline de execução melhora a eficiência do processo, tornando-o mais rápido e eficaz. Este estudo fornece uma base sólida para o desenvolvimento de soluções de Machine Learning aplicadas à gestão da poluição plástica.
