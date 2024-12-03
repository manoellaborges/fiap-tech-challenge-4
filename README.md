# Previsão de Preços de Ações com Modelo LSTM

## Índice

- [Previsão de Preços de Ações com Modelo LSTM](#previsão-de-preços-de-ações-com-modelo-lstm)
  - [Índice](#índice)
  - [Visão Geral](#visão-geral)
    - [Principais Funcionalidades](#principais-funcionalidades)
    - [Acesso à Documentação Interativa](#acesso-à-documentação-interativa)
  - [Pré-requisitos](#pré-requisitos)
  - [Configuração do Ambiente](#configuração-do-ambiente)
    - [Instalação do Docker](#instalação-do-docker)
    - [Instalação do Python](#instalação-do-python)
    - [Instruções para Construção e Execução](#instruções-para-construção-e-execução)
      - [Construir a Imagem Docker](#construir-a-imagem-docker)
    - [Executar o Contêiner Docker](#executar-o-contêiner-docker)
  - [Uso da API](#uso-da-api)
    - [API de Treino (train)](#api-de-treino-train)
      - [Endpoint - Treino](#endpoint---treino)
      - [Descrição - Treino](#descrição---treino)
      - [Exemplo de Requisição - Treino](#exemplo-de-requisição---treino)
      - [Exemplo de Resposta - Treino](#exemplo-de-resposta---treino)
    - [API de Predição (predict)](#api-de-predição-predict)
      - [Endpoint - Predict](#endpoint---predict)
      - [Descrição - Predict](#descrição---predict)
      - [Funcionalidade](#funcionalidade)
    - [Parâmetros](#parâmetros)
      - [Exemplo de Requisição - Predict](#exemplo-de-requisição---predict)
      - [Exemplo de Resposta - Predict](#exemplo-de-resposta---predict)

## Visão Geral

Bem-vindo à documentação da API de Previsão de Preços de Ações, desenvolvida como parte de um desafio tecnológico da faculdade FIAP. Esta solução utiliza redes neurais recorrentes LSTM (Long Short-Term Memory) para prever preços futuros de ações, demonstrando o potencial das tecnologias de aprendizado de máquina em aplicações financeiras.

### Principais Funcionalidades

- **Previsões Precisas**: Utiliza um modelo LSTM treinado com dados históricos de preços de ações para fornecer estimativas confiáveis.
- **Entrada Flexível**: Oferece a possibilidade de prever preços usando apenas o símbolo de uma ação ou fornecendo dados históricos personalizados para mais controle.
- **Validação e Segurança de Dados**: Inclui verificações automáticas para garantir a qualidade dos dados de entrada, como detecção de outliers e validação mínima de dados.
- **Integração Simples**: Fácil de integrar em aplicações existentes, permitindo previsões em tempo real que podem informar decisões de investimento estratégicas.

### Acesso à Documentação Interativa

Para explorar e testar a API interativamente, acesse o Swagger através do link: [Swagger UI](http://localhost:8000/docs). Esta interface fornece uma maneira amigável de visualizar todos os endpoints disponíveis, testar requisições, e entender os parâmetros e respostas esperados.

## Pré-requisitos

- **Python 3.10 ou superior**: O projeto utiliza funcionalidades compatíveis com esta versão.
- **Docker**: Para construção e execução de contêineres.
- **Docker Compose** (opcional): Para gerenciamento de múltiplos contêineres.

## Configuração do Ambiente

### Instalação do Docker

Para instalar o Docker, siga as instruções do [site oficial do Docker](https://www.docker.com/products/docker-desktop).

### Instalação do Python

Recomenda-se o uso do `pyenv` para gerenciar versões de Python. Instale o `pyenv` e configure a versão correta do Python:

```bash
pyenv install 3.10.2
pyenv local 3.10.2
```

### Instruções para Construção e Execução

#### Construir a Imagem Docker

Navegue até o diretório raiz do projeto (onde o Dockerfile está localizado) e execute:

```bash
docker build -t meu-app-fastapi .
```

### Executar o Contêiner Docker

Para iniciar o contêiner e expor a API:

```bash
docker run -p 8000:8000 meu-app-fastapi
```

A API estará acessível em `http://localhost:8000`.

## Uso da API

### API de Treino (train)

#### Endpoint - Treino

`POST /train/AAPL`

#### Descrição - Treino

A API `train` treina um modelo de rede neural LSTM utilizando dados históricos de preços de ações para prever preços futuros.

#### Exemplo de Requisição - Treino

```bash
curl --request POST \
  --url http://localhost:8000/train/AAPL
```

#### Exemplo de Resposta - Treino

```json
{
  "message": "Modelo treinado com sucesso."
}
```

### API de Predição (predict)

#### Endpoint - Predict

`POST /predict`

#### Descrição - Predict

 A API `predict` é projetada para utilizar um modelo de rede neural LSTM previamente treinado com dados históricos de preços de ações para realizar previsões sobre preços futuros.

#### Funcionalidade

- **Entrada Flexível**: Aceita tanto um símbolo de ação quanto uma lista de dados históricos (últimos 60 dias) personalizados.
- **Validação de Dados**: Realiza validações básicas nos dados personalizados, incluindo verificação de quantidade mínima de dados e detecção de outliers.
- **Previsão de Preços**: Retorna o preço previsto para o próximo intervalo de tempo com base nos dados fornecidos.

### Parâmetros

- **symbol**: O símbolo da ação para a qual se deseja prever o preço. A API buscará automaticamente os dados históricos necessários.

- **custom_data**: (opcional) Uma lista de valores numéricos representando dados históricos personalizados para serem usados na previsão. Se fornecido, deve conter pelo menos 60 pontos de dados.

#### Exemplo de Requisição - Predict

```bash
curl --request POST \
  --url 'http://localhost:8000/predict' \
  --data '{
  "symbol": "AAPL"
}'
```

```bash
curl --request POST \
  --url 'http://localhost:8000/predict' \
  --data '{
  "symbol": "AAPL",
  "custom_data": [135.67, 136.98, 137.50, 134.56, ..., 138.70, 136.50]
}'
```

#### Exemplo de Resposta - Predict

```json
{
  "predicted_price": 179.36
}
```
