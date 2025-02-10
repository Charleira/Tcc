package stockservices

import (
	stock_model "stock.api/models/stock_model"

	"github.com/jinzhu/gorm"
)

// AddStock insere um novo stock no banco
func AddStock(db *gorm.DB, stock stock_model.Stock) error {
	return db.Create(&stock).Error
}

// GetStocks retorna todos os stocks
func GetStocks(db *gorm.DB) ([]stock_model.Stock, error) {
	var stocks []stock_model.Stock
	if err := db.Find(&stocks).Error; err != nil {
		return nil, err
	}
	return stocks, nil
}
