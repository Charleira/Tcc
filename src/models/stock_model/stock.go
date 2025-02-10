package stockmodels

import "time"

// Stock representa a tabela de ações
type Stock struct {
	ID            uint      `json:"id"`
	Ticker        string    `json:"ticker"`
	Date          time.Time `json:"date"`
	Close         float64   `json:"close"`
	Sma10         float64   `json:"sma_10"`
	Sma20         float64   `json:"sma_20"`
	Rsi           float64   `json:"rsi"`
	BollingerHigh float64   `json:"bollinger_high"`
	BollingerLow  float64   `json:"bollinger_low"`
	CreatedAt     time.Time `json:"created_at"`
}
