package stockcontroller

import (
	"github.com/go-fuego/fuego"
	"github.com/go-fuego/fuego/option"
	"github.com/go-fuego/fuego/param"
	"github.com/jinzhu/gorm"
	stock_model "stock.api/models/stock_model"
	stock_service "stock.api/services/stock_service"
)

// StockResources estrutura que agrupa os handlers relacionados a stocks
type StockResources struct {
	DB *gorm.DB
}

// CreateStock insere um novo estoque no banco
func (rs StockResources) CreateStock(c fuego.ContextWithBody[stock_model.Stock]) (map[string]string, error) {
	var stock stock_model.Stock
	body, err := c.Body()
	if err != nil {
		return nil, err
	}

	stock = body

	// Passando o banco de dados como parâmetro para AddStock
	if err := stock_service.AddStock(rs.DB, stock); err != nil {
		return nil, err
	}

	return map[string]string{"message": "Stock added successfully"}, nil
}

// GetAllStocks retorna todos os stocks
func (rs StockResources) GetAllStocks(c fuego.ContextNoBody) ([]stock_model.Stock, error) {
	// Passando o banco de dados como parâmetro para GetStocks
	stocks, err := stock_service.GetStocks(rs.DB)
	if err != nil {
		return nil, err
	}
	return stocks, nil
}

// Routes configura as rotas para os stocks
func (rs StockResources) Routes(s *fuego.Server) {
	// Cria o grupo de rotas "/stocks"
	stocksGroup := fuego.Group(s, "/stocks")

	// Definir a rota POST /stocks para adicionar um estoque
	fuego.Post(stocksGroup, "/", rs.CreateStock,
		option.Description("Add a new stock entry"),
		option.RequestContentType("application/json"),
	)

	// Definir a rota GET /stocks para retornar todos os stocks
	fuego.Get(stocksGroup, "/", rs.GetAllStocks,
		option.Description("Get all stock entries"),
		option.Query("page", "Page number", param.Default(1)),
	)
}
