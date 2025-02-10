package news_service

import (
	"github.com/jinzhu/gorm"
	news_model "stock.api/models/news_model"
)

// AddNews insere uma nova notícia no banco de dados
func AddNews(db *gorm.DB, news news_model.News) error {
	return db.Create(&news).Error
}

// GetNews retorna todas as notícias
func GetNews(db *gorm.DB) ([]news_model.News, error) {
	var news []news_model.News
	if err := db.Find(&news).Error; err != nil {
		return nil, err
	}
	return news, nil
}
