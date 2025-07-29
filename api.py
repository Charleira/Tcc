import os
import time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from newspaper import Article
from textblob import TextBlob

# ----- Configs -----
API_KEY = os.getenv("FINNHUB_API_KEY", "cul25nhr01qqav2uqppgcul25nhr01qqav2uqpq0")
FINNHUB_URL = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={API_KEY}"
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_lstm_multivariado.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obter_sentimento_noticias(ticker, api_key):
    fim = int(time.time())
    inicio = fim - (30 * 24 * 60 * 60)  # últimos 30 dias em segundos

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}"
        f"&from={datetime.fromtimestamp(inicio).strftime('%Y-%m-%d')}"
        f"&to={datetime.fromtimestamp(fim).strftime('%Y-%m-%d')}"
        f"&token={api_key}"
    )

    try:
        resp = requests.get(url, timeout=15)
        noticias = resp.json()
    except Exception:
        return 0.0

    sentimentos = []

    for noticia in noticias[:10]:  # Limita para evitar travamento
        score = None

        # 🟢 1. Tenta usar o score da própria API
        if 'sentiment' in noticia and isinstance(noticia['sentiment'], dict) and 'score' in noticia['sentiment']:
            score = noticia['sentiment']['score']

        # 🟡 2. Se não tiver score, tenta análise via scraping
        if score is None and noticia.get("url"):
            try:
                artigo = Article(noticia["url"])
                artigo.download()
                artigo.parse()

                texto = artigo.text
                analise = TextBlob(texto)
                score = analise.sentiment.polarity  # -1 a 1
            except Exception:
                continue  # Ignora se erro no scraping

        if score is not None:
            sentimentos.append(score)

    if sentimentos:
        return float(np.mean(sentimentos))  # média do sentimento
    return 0.0

def obter_dados_finnhub(ticker, api_key):
    fim = datetime.now()
    inicio = fim - timedelta(days=2*365)

    df = yf.download(ticker, start=inicio.strftime('%Y-%m-%d'), end=fim.strftime('%Y-%m-%d'), auto_adjust=False)

    if df.empty:
        raise ValueError(f"Sem dados do Yahoo Finance para {ticker}")

    df = df[["Close"]]
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Bollinger_Middle"] = df["Close"].rolling(window=20).mean()
    df["Bollinger_STD"] = df["Close"].rolling(window=20).std()
    df["Bollinger_High"] = df["Bollinger_Middle"] + 2 * df["Bollinger_STD"]
    df["Bollinger_Low"] = df["Bollinger_Middle"] - 2 * df["Bollinger_STD"]

    sentimento = obter_sentimento_noticias(ticker, api_key)
    df["Sentiment"] = sentimento

    df.dropna(inplace=True)
    return df

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(100, 100, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(100, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        return x

def prever_para_7_dias(ticker, model, seq_length=60, dias=7):
    
    from sklearn.preprocessing import MinMaxScaler

    df = obter_dados_finnhub(ticker, API_KEY)
    cols = ["Close", "SMA_10", "SMA_20", "Bollinger_Low", "Bollinger_High", "Sentiment"]
    dados = df[cols].values

    # Scaler multivariado para a entrada do modelo
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados)

    # Scaler só para Close (para inverter a previsão)
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[["Close"]].values)

    input_seq = dados_normalizados[-seq_length:].copy()

    previsoes_normalizadas = []
    model.eval()
    with torch.no_grad():
        for _ in range(dias):
            X_input = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_norm = model(X_input).cpu().numpy()[0][0]
            previsoes_normalizadas.append(pred_norm)

            # Mantém as demais features da última linha (exceto o Close prevista)
            nova_linha = input_seq[-1].copy()
            nova_linha[0] = pred_norm
            input_seq = np.vstack([input_seq[1:], nova_linha])

    previsoes_reais = []
    for p in previsoes_normalizadas:
        inv_close = scaler_close.inverse_transform([[p]])[0][0]
        previsoes_reais.append(float(inv_close))

    # Datas (simples: dias corridos; ajuste para dias úteis se quiser)
    last_date = df.index[-1]
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, dias + 1)]

    valor_atual = float(df["Close"].iloc[-1])
    return valor_atual, future_dates, previsoes_reais

# ======== CARREGA O MODELO TREINADO (.pth) ========
INPUT_SIZE = 6
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Modelo .pth não encontrado. Treine e salve o modelo antes de subir a API."
    )

model = LSTMModel(INPUT_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======== FASTAPI ========
app = FastAPI(title="Stock Forecast API", version="1.0.0")

class PredictRequest(BaseModel):
    ticker: str
    days: int = 7

class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    days: int
    dates: List[str]
    predictions: List[float]


@app.get("/historico/{ticker}")
def historico_acao(
    ticker: str,
):
    try:
        fim = datetime.now()
        inicio = fim - timedelta(days=30)
        df = yf.download(
            ticker.upper(),
            start=inicio.strftime('%Y-%m-%d'),
            end=fim.strftime('%Y-%m-%d'),
            interval="1d",
            auto_adjust=True
        )
        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker não encontrado ou sem dados.")

        # Preços diários
        historico_30d = [
            {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"].iloc[0])}
            for idx, row in df.tail(30).iterrows()
        ]
        historico_15d = [
            {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"].iloc[0])}
            for idx, row in df.tail(15).iterrows()
        ]
        historico_7d = [
            {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"].iloc[0])}
            for idx, row in df.tail(7).iterrows()
        ]

        # Preços por hora do último dia
        ultimo_dia = df.index[-1].strftime("%Y-%m-%d")
        df_hora = yf.download(
            ticker.upper(),
            start=ultimo_dia,
            end=(fim + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval="60m",
            auto_adjust=True
        )
        historico_1d_horas = [
            {"datetime": idx.strftime("%Y-%m-%d %H:%M"), "close": float(row["Close"].iloc[0])}
            for idx, row in df_hora.iterrows()
        ]

        return {
            "ticker": ticker.upper(),
            "historico_30d": historico_30d,
            "historico_15d": historico_15d,
            "historico_7d": historico_7d,
            "historico_1d_horas": historico_1d_horas
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar histórico: {str(e)}")

@app.get("/noticias-sentimento/{ticker}")
def obter_noticias_com_sentimento(ticker: str):
    fim = int(time.time())
    inicio = fim - (30 * 24 * 60 * 60)  # últimos 30 dias

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker.upper()}"
        f"&from={datetime.fromtimestamp(inicio).strftime('%Y-%m-%d')}"
        f"&to={datetime.fromtimestamp(fim).strftime('%Y-%m-%d')}"
        f"&token={API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        noticias = resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar notícias: {str(e)}")

    resultado = []
    for n in noticias[:5]:  # Limita para 5 para evitar lentidão
        url_noticia = n.get("url")
        texto = ""
        sentimento = "indefinido"

        # Extrai o conteúdo da notícia (texto completo)
        try:
            artigo = Article(url_noticia)
            artigo.download()
            artigo.parse()
            texto = artigo.text

            # Analisa o sentimento do texto com TextBlob
            analise = TextBlob(texto)
            polaridade = analise.sentiment.polarity  # -1 (negativo) a 1 (positivo)

            if polaridade > 0.1:
                sentimento = "positivo"
            elif polaridade < -0.1:
                sentimento = "negativo"
            else:
                sentimento = "neutro"

        except Exception:
            sentimento = "erro_ao_processar"

        resultado.append({
            "headline": n.get("headline"),
            "datetime": datetime.fromtimestamp(n.get("datetime")).strftime("%Y-%m-%d %H:%M"),
            "url": url_noticia,
            "sentimento": sentimento
        })

    return {
        "ticker": ticker.upper(),
        "total": len(resultado),
        "noticias": resultado
    }

@app.get("/tickers")
def get_tickers():
    try:
        response = requests.get(FINNHUB_URL, timeout=10)
        response.raise_for_status()
        tickers_data = response.json()
        tickers = [ticker["symbol"] for ticker in tickers_data]
        return {"count": len(tickers), "tickers": tickers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        current_price, dates, preds = prever_para_7_dias(
            ticker=req.ticker.upper(),
            model=model,
            seq_length=60,
            dias=req.days
        )
        return PredictResponse(
            ticker=req.ticker.upper(),
            current_price=current_price,
            days=req.days,
            dates=dates,
            predictions=preds
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
