package main

import (
	"log"

	routes "stock.api/controllers"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/mysql"
)

func main() {
	// Conectar ao banco de dados (substitua com suas credenciais)
	db, err := gorm.Open("mysql", "user:password@/dbname?charset=utf8&parseTime=True&loc=Local")
	if err != nil {
		log.Fatal("Erro ao conectar no banco:", err)
	}
	defer db.Close()

	// Inicializar servidor Fuego
	server := routes.SetupRoutes(db)

	// Configurar e rodar o servidor
	server.Run()
}
