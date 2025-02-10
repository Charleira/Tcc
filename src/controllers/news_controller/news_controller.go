package newscontroller

import (
	"github.com/go-fuego/fuego"
	"github.com/go-fuego/fuego/option"
	"github.com/go-fuego/fuego/param"
	"github.com/jinzhu/gorm"
	news_model "stock.api/models/news_model"
	news_service "stock.api/services/news_service"
)

// NewsResources estrutura que agrupa os handlers relacionados a notícias
type NewsResources struct {
	DB *gorm.DB
}

// AddNews insere uma nova notícia
func (rs NewsResources) AddNews(c fuego.ContextWithBody[news_model.News]) (map[string]string, error) {
	var news news_model.News
	body, err := c.Body()
	if err != nil {
		return nil, err
	}

	news = body

	// Passando o banco de dados como parâmetro para AddNews
	if err := news_service.AddNews(rs.DB, news); err != nil {
		return nil, err
	}

	return map[string]string{"message": "News added successfully"}, nil
}

// GetAllNews retorna todas as notícias
func (rs NewsResources) GetAllNews(c fuego.ContextNoBody) ([]news_model.News, error) {
	// Passando o banco de dados como parâmetro para GetNews
	news, err := news_service.GetNews(rs.DB)
	if err != nil {
		return nil, err
	}
	return news, nil
}

// Routes configura as rotas para as notícias
func (rs NewsResources) Routes(s *fuego.Server) {
	// Cria o grupo de rotas "/news"
	newsGroup := fuego.Group(s, "/news")

	// Definir a rota POST /news para adicionar uma notícia
	fuego.Post(newsGroup, "/", rs.AddNews,
		option.Description("Add a new news article"),
		option.RequestContentType("application/json"),
		option.ResponseHeader("X-Header", "header description"),
	)

	// Definir a rota GET /news para retornar todas as notícias
	fuego.Get(newsGroup, "/", rs.GetAllNews,
		option.Description("Get all news articles"),
		option.Query("page", "Page number", param.Default(1)),
	)
}
