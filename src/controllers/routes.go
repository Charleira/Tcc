package controllers

import (
	"github.com/go-fuego/fuego"
	"github.com/jinzhu/gorm"
	news "stock.api/controllers/news_controller"
	stocks "stock.api/controllers/stocks_controller"
)

// SetupRoutes configura as rotas da API
func SetupRoutes(db *gorm.DB) *fuego.Server {
	server := fuego.NewServer()

	// Registrar as rotas de ações (stocks)
	stocksResources := stocks.StockResources{DB: db}
	stocksResources.Routes(server) // Chama a função Routes que já está configurada no controlador

	// Registrar as rotas de notícias (news)
	newsResources := news.NewsResources{}
	newsResources.Routes(server) // Chama a função Routes que já está configurada no controlador

	return server
}
