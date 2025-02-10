package newsmodels

import "time"

// News representa a tabela de análise de sentimentos das notícias
type News struct {
	ID           uint      `json:"id"`
	Ticker       string    `json:"ticker"`
	Date         time.Time `json:"date"`
	Sentiment    string    `json:"sentiment"`
	ArticleTitle string    `json:"article_title"`
	CreatedAt    time.Time `json:"created_at"`
}
