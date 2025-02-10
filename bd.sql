CREATE TABLE stocks_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    close FLOAT NOT NULL,
    sma_10 FLOAT NOT NULL,
    sma_20 FLOAT NOT NULL,
    rsi FLOAT NOT NULL,
    bollinger_high FLOAT NOT NULL,
    bollinger_low FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE news_sentiment_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    article_title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
