# %%
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from newspaper import Article
from textblob import TextBlob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

api_key = "cul25nhr01qqav2uqppgcul25nhr01qqav2uqpq0"

url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key}"
response = requests.get(url)
tickers_data = response.json()
tickers = [ticker["symbol"] for ticker in tickers_data]

def obter_sentimento_noticias(ticker, api_key):
    fim = int(time.time())
    inicio = fim - (2 * 365 * 24 * 60 * 60)  # 2 anos em segundos

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}"
        f"&from={datetime.fromtimestamp(inicio).strftime('%Y-%m-%d')}"
        f"&to={datetime.fromtimestamp(fim).strftime('%Y-%m-%d')}"
        f"&token={api_key}"
    )

    resp = requests.get(url)
    noticias = resp.json()

    sentimentos = []
    for noticia in noticias:
        if 'sentiment' in noticia and isinstance(noticia['sentiment'], dict) and 'score' in noticia['sentiment']:
            sentimentos.append(noticia['sentiment']['score'])

    if sentimentos:
        return np.mean(sentimentos)
    return 0.0

def obter_dados_finnhub(ticker, api_key):
    fim = datetime.now()
    inicio = fim - timedelta(days=2*365)

    df = yf.download(ticker, start=inicio.strftime('%Y-%m-%d'), end=fim.strftime('%Y-%m-%d'), auto_adjust=False)

    if df.empty:
        print(f"⚠️ Aviso: dados vazios para ticker '{ticker}'. Pulando.")
        return None

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

def criar_sequencias(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

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

modelo_path = "modelo_lstm_multivariado.pth"
scaler_path = "scaler.save"
scaler_close_path = "scaler_close.save"

if os.path.exists(modelo_path) and os.path.exists(scaler_path) and os.path.exists(scaler_close_path):
    print(f"✅ Carregando modelo e scalers salvos...")
    input_size = 6
    model = LSTMModel(input_size).to(device)
    model.load_state_dict(torch.load(modelo_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    scaler_close = joblib.load(scaler_close_path)
else:
    print(f"🚀 Treinando novo modelo...")
    primeiro_ticker = tickers[0]
    df = obter_dados_finnhub(primeiro_ticker, api_key)

    dados_para_modelo = df[["Close", "SMA_10", "SMA_20", "Bollinger_Low", "Bollinger_High", "Sentiment"]].values

    # scaler multivariado para todas as features
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados_para_modelo)
    joblib.dump(scaler, scaler_path)  # salva scaler multivariado

    # scaler exclusivo para Close, só para inverter previsões corretamente
    scaler_close = MinMaxScaler()
    close_values = df[["Close"]].values
    close_normalized = scaler_close.fit_transform(close_values)
    joblib.dump(scaler_close, scaler_close_path)

    seq_length = 60
    X, y = criar_sequencias(dados_normalizados, seq_length)

    X_treino = torch.tensor(X[:int(len(X)*0.75)], dtype=torch.float32).to(device)
    y_treino = torch.tensor(y[:int(len(X)*0.75)], dtype=torch.float32).unsqueeze(1).to(device)
    X_valid = torch.tensor(X[int(len(X)*0.75):], dtype=torch.float32).to(device)
    y_valid = torch.tensor(y[int(len(X)*0.75):], dtype=torch.float32).unsqueeze(1).to(device)

    model = LSTMModel(X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_treino)
        loss = criterion(output, y_treino)
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = criterion(model(X_valid), y_valid)
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), modelo_path)
    print(f"✅ Modelo salvo como '{modelo_path}'!")

def prever_para_7_dias(ticker, model, seq_length=60, dias=7):
    df = obter_dados_finnhub(ticker, api_key)
    dados = df[["Close", "SMA_10", "SMA_20", "Bollinger_Low", "Bollinger_High", "Sentiment"]].values

    # Ajusta scaler para os dados do ticker atual
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados)

    scaler_close = MinMaxScaler()
    close_values = df[["Close"]].values
    scaler_close.fit(close_values)

    input_seq = dados_normalizados[-seq_length:].copy()

    previsoes_normalizadas = []
    model.eval()
    with torch.no_grad():
        for _ in range(dias):
            X_input = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_norm = model(X_input).cpu().numpy()[0][0]
            previsoes_normalizadas.append(pred_norm)

            nova_linha = input_seq[-1].copy()
            nova_linha[0] = pred_norm
            input_seq = np.vstack([input_seq[1:], nova_linha])

    previsoes_reais = []
    for p in previsoes_normalizadas:
        inv_close = scaler_close.inverse_transform([[p]])[0][0]
        previsoes_reais.append(inv_close)

    return previsoes_reais


print("📈 Ações disponíveis para previsão:")
for i, ticker in enumerate(tickers[:50]):
    print(f"{i + 1}. {ticker}")

while True:
    ticker_escolhido = input("Digite o código da ação que deseja prever: ").upper()
    if ticker_escolhido in tickers:
        break
    print("❌ Código inválido! Digite um ticker da lista.")

try:
    print(f"🔄 Buscando dados para {ticker_escolhido}...")
    df_atual = obter_dados_finnhub(ticker_escolhido, api_key)
    valor_atual = float(df_atual["Close"].iloc[-1])
    print(f"Valor atual da ação {ticker_escolhido}: ${valor_atual:.2f}")

    previsoes_7_dias = prever_para_7_dias(ticker_escolhido, model)

    for i, preco in enumerate(previsoes_7_dias, 1):
        print(f"Previsão para o dia {i}: ${preco:.2f}")

    previsao_df = pd.DataFrame({
        "Dia": [f"Dia {i}" for i in range(1, len(previsoes_7_dias)+1)],
        "Ticker": ticker_escolhido,
        "Previsão": previsoes_7_dias
    })

    previsao_df.to_csv(f"previsao_{ticker_escolhido}_7dias.csv", index=False)
    print(f"✅ Previsões salvas em 'previsao_{ticker_escolhido}_7dias.csv'!")

except Exception as e:
    print(f"⚠️ Erro ao prever: {e}")


# %%
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib  # para salvar e carregar o scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

api_key = "cul25nhr01qqav2uqppgcul25nhr01qqav2uqpq0"

url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key}"
response = requests.get(url)
tickers_data = response.json()
tickers = [ticker["symbol"] for ticker in tickers_data]

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
        print(f"⚠️ Aviso: dados vazios para ticker '{ticker}'. Pulando.")
        return None

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

def criar_sequencias(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

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

modelo_path = "modelo_lstm_multivariado.pth"
scaler_path = "scaler.save"
scaler_close_path = "scaler_close.save"

if os.path.exists(modelo_path) and os.path.exists(scaler_path) and os.path.exists(scaler_close_path):
    print(f"✅ Carregando modelo e scalers salvos...")
    input_size = 6
    model = LSTMModel(input_size).to(device)
    model.load_state_dict(torch.load(modelo_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    scaler_close = joblib.load(scaler_close_path)
else:
    print(f"🚀 Treinando novo modelo...")
    primeiro_ticker = tickers[0]
    df = obter_dados_finnhub(primeiro_ticker, api_key)

    dados_para_modelo = df[["Close", "SMA_10", "SMA_20", "Bollinger_Low", "Bollinger_High", "Sentiment"]].values

    # scaler multivariado para todas as features
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados_para_modelo)
    joblib.dump(scaler, scaler_path)  # salva scaler multivariado

    # scaler exclusivo para Close, só para inverter previsões corretamente
    scaler_close = MinMaxScaler()
    close_values = df[["Close"]].values
    close_normalized = scaler_close.fit_transform(close_values)
    joblib.dump(scaler_close, scaler_close_path)

    seq_length = 60
    X, y = criar_sequencias(dados_normalizados, seq_length)

    X_treino = torch.tensor(X[:int(len(X)*0.75)], dtype=torch.float32).to(device)
    y_treino = torch.tensor(y[:int(len(X)*0.75)], dtype=torch.float32).unsqueeze(1).to(device)
    X_valid = torch.tensor(X[int(len(X)*0.75):], dtype=torch.float32).to(device)
    y_valid = torch.tensor(y[int(len(X)*0.75):], dtype=torch.float32).unsqueeze(1).to(device)

    model = LSTMModel(X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_treino)
        loss = criterion(output, y_treino)
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = criterion(model(X_valid), y_valid)
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), modelo_path)
    print(f"✅ Modelo salvo como '{modelo_path}'!")

def prever_para_7_dias(ticker, model, seq_length=60, dias=7):
    df = obter_dados_finnhub(ticker, api_key)
    dados = df[["Close", "SMA_10", "SMA_20", "Bollinger_Low", "Bollinger_High", "Sentiment"]].values

    # Ajusta scaler para os dados do ticker atual
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados)

    scaler_close = MinMaxScaler()
    close_values = df[["Close"]].values
    scaler_close.fit(close_values)

    input_seq = dados_normalizados[-seq_length:].copy()

    previsoes_normalizadas = []
    model.eval()
    with torch.no_grad():
        for _ in range(dias):
            X_input = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred_norm = model(X_input).cpu().numpy()[0][0]
            previsoes_normalizadas.append(pred_norm)

            nova_linha = input_seq[-1].copy()
            nova_linha[0] = pred_norm
            input_seq = np.vstack([input_seq[1:], nova_linha])

    previsoes_reais = []
    for p in previsoes_normalizadas:
        inv_close = scaler_close.inverse_transform([[p]])[0][0]
        previsoes_reais.append(inv_close)

    return previsoes_reais


print("📈 Ações disponíveis para previsão:")
for i, ticker in enumerate(tickers[:50]):
    print(f"{i + 1}. {ticker}")

while True:
    ticker_escolhido = input("Digite o código da ação que deseja prever: ").upper()
    if ticker_escolhido in tickers:
        break
    print("❌ Código inválido! Digite um ticker da lista.")

try:
    print(f"🔄 Buscando dados para {ticker_escolhido}...")
    df_atual = obter_dados_finnhub(ticker_escolhido, api_key)
    valor_atual = float(df_atual["Close"].iloc[-1])
    print(f"Valor atual da ação {ticker_escolhido}: ${valor_atual:.2f}")

    previsoes_7_dias = prever_para_7_dias(ticker_escolhido, model)

    for i, preco in enumerate(previsoes_7_dias, 1):
        print(f"Previsão para o dia {i}: ${preco:.2f}")

    previsao_df = pd.DataFrame({
        "Dia": [f"Dia {i}" for i in range(1, len(previsoes_7_dias)+1)],
        "Ticker": ticker_escolhido,
        "Previsão": previsoes_7_dias
    })

    previsao_df.to_csv(f"previsao_{ticker_escolhido}_7dias.csv", index=False)
    print(f"✅ Previsões salvas em 'previsao_{ticker_escolhido}_7dias.csv'!")

except Exception as e:
    print(f"⚠️ Erro ao prever: {e}")