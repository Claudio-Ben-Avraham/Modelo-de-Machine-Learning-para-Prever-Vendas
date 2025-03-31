Modelo de Machine Learning para Prever Vendas de Sorvetes - Gelato Mágico
Descrição

Este projeto visa desenvolver um modelo de Machine Learning para prever a quantidade de sorvetes que serão vendidos em uma sorveteria chamada Gelato Mágico, com base na temperatura ambiente. A ideia é usar regressão preditiva para antecipar a demanda e otimizar a produção, evitando o desperdício de recursos ou a perda de vendas devido à falta de estoque.
Objetivos do Projeto

✅ Treinar um modelo de Machine Learning para prever as vendas de sorvetes com base na temperatura do dia.
✅ Registrar e gerenciar o modelo utilizando a ferramenta MLflow, facilitando o rastreamento e versionamento do modelo.
✅ Implementar o modelo para previsões em tempo real em um ambiente de cloud computing.
✅ Criar um pipeline estruturado para treinar e testar o modelo, garantindo a reprodutibilidade do processo de desenvolvimento.
Tecnologias Utilizadas

    Python: Linguagem de programação principal para construção do modelo.

    Scikit-learn: Biblioteca para Machine Learning, utilizada para implementar o modelo de regressão.

    MLflow: Plataforma para gerenciar o ciclo de vida do modelo, incluindo experimentos, treinamento e deploy.

    Pandas: Biblioteca para manipulação e análise de dados.

    Matplotlib/Seaborn: Bibliotecas para visualização dos dados e resultados.

    Azure Machine Learning: Plataforma de computação em nuvem para treinar, testar e implantar o modelo.

Estrutura do Repositório

    data/: Diretório contendo os dados de histórico de vendas e temperatura.

    notebooks/: Jupyter notebooks para exploração dos dados, treinamento e validação do modelo.

    src/: Código fonte que contém os scripts principais para treinamento, teste e deploy do modelo.

    models/: Diretório para armazenar o modelo treinado.

    requirements.txt: Arquivo com as dependências necessárias para rodar o projeto.

    README.md: Este arquivo com a descrição do projeto.

Como Rodar o Projeto
1. Preparar o Ambiente

Certifique-se de ter Python 3.x instalado e configure um ambiente virtual:

python -m venv venv
source venv/bin/activate  # Para Linux/Mac
venv\Scripts\activate     # Para Windows

2. Instalar Dependências

Instale as dependências necessárias com o seguinte comando:

pip install -r requirements.txt

3. Carregar e Explorar os Dados

Acesse o diretório notebooks/ e explore os notebooks para analisar os dados históricos de temperatura e vendas. Utilize o Pandas para carregar os dados e o Matplotlib ou Seaborn para visualizações.
4. Treinar o Modelo

Treine o modelo de regressão preditiva utilizando a biblioteca Scikit-learn. O modelo pode ser um regressor linear, por exemplo, para prever a quantidade de sorvetes vendidos com base na temperatura do dia.

Exemplo de código para treinar o modelo:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Carregar os dados
data = pd.read_csv("data/sorvete_temperatura.csv")

# Dividir os dados em variáveis independentes e dependentes
X = data["temperatura"].values.reshape(-1, 1)
y = data["vendas"].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, predictions)
print(f"Erro quadrático médio: {mse}")

5. Registrar o Modelo com MLflow

Utilize o MLflow para registrar o modelo, permitindo gerenciar diferentes versões do modelo e suas métricas.

Exemplo de código para registrar o modelo com MLflow:

import mlflow
import mlflow.sklearn

# Iniciar o registro do experimento no MLflow
with mlflow.start_run():
    # Registrar parâmetros
    mlflow.log_param("modelo", "Regressão Linear")
    mlflow.log_param("test_size", 0.2)
    
    # Registrar o modelo treinado
    mlflow.sklearn.log_model(model, "modelo_sorvete")
    
    # Registrar métricas
    mlflow.log_metric("mse", mse)

6. Implantação e Previsões em Tempo Real

O modelo pode ser implantado em um serviço de cloud computing (ex: Azure Machine Learning). Para previsões em tempo real, você pode configurar uma API REST que consome a entrada de temperatura e retorna a previsão de vendas de sorvete.

Exemplo básico de API com Flask para previsões em tempo real:

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregar o modelo treinado
model = joblib.load("models/modelo_sorvete.pkl")

@app.route('/prever', methods=['POST'])
def prever_vendas():
    data = request.get_json()  # Espera-se que a entrada seja um JSON com a temperatura
    temperatura = data["temperatura"]
    
    # Prever as vendas
    previsao = model.predict([[temperatura]])[0]
    
    return jsonify({"vendas_previstas": previsao})

if __name__ == '__main__':
    app.run(debug=True)

7. Agendamento e Pipeline

Você pode configurar um pipeline no Azure Machine Learning para agendar o treinamento do modelo e atualizações periódicas com novos dados. Isso garante a reprodutibilidade e automação do processo.
