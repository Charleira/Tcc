# Stock Forecast API

API para consulta de histórico, previsão de preços e análise de sentimento de ações utilizando FastAPI, PyTorch e Yahoo Finance.

## Funcionalidades

- Histórico de preços (30, 15, 7 dias e por hora do último dia)
- Previsão de preços com LSTM
- Análise de sentimento de notícias
- Listagem de tickers disponíveis

## Requisitos

- Python 3.9+
- Docker (opcional, para rodar em container)

## Instalação Local

1. Clone o repositório:
    ```sh
    git clone https://github.com/seuusuario/seurepo.git
    cd seurepo
    ```

2. Crie e ative um ambiente virtual:
    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```

3. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

4. Adicione o arquivo do modelo `.pth` treinado na raiz do projeto (ou ajuste o caminho em `MODEL_PATH`).

5. Exporte a variável de ambiente da chave da API Finnhub (opcional):
    ```sh
    set FINNHUB_API_KEY=sua_chave_aqui
    ```

6. Rode a API:
    ```sh
    uvicorn api:app --reload
    ```

Acesse em [http://localhost:8000/docs](http://localhost:8000/docs) para testar os endpoints.

---

## Rodando com Docker

1. Crie o arquivo `Dockerfile` na raiz do projeto:

    ```dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    EXPOSE 8000

    ENV MODEL_PATH=modelo_lstm_multivariado.pth

    CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

2. Construa a imagem Docker:
    ```sh
    docker build -t stock-forecast-api .
    ```

3. Rode o container:
    ```sh
    docker run -d -p 8000:8000 -e FINNHUB_API_KEY=sua_chave_aqui stock-forecast-api
    ```

Acesse em [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Observações

- Certifique-se de ter o arquivo do modelo `.pth` no diretório correto.
- Para usar GPU no Docker, utilize uma imagem compatível e configure o driver CUDA.
- Dependências principais: FastAPI, yfinance, torch, pandas, numpy, requests, newspaper3k, textblob.

---

## Exemplos de uso

- `GET /historico/{ticker}`: Histórico de preços para gráficos.
- `POST /predict`: Previsão de preços futuros.
- `GET /noticias-sentimento/{ticker}`: Notícias recentes com análise de sentimento.

---