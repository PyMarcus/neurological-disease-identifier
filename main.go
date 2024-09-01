package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
)

type Classification struct {
	Classification string `json: "classification"`
}

func main() {
	r := gin.Default()
	r.LoadHTMLGlob("templates/*.html")
	r.Static("/assets", "./templates/assets")
	r.GET("/", func(c *gin.Context) {
		c.File("templates/index.html")
	})

	r.GET("/result", func(c *gin.Context) {
		c.HTML(http.StatusOK, "result.html", gin.H{
			"message": c.Query("message"),
		})
	})

	r.POST("/upload", func(ctx *gin.Context) {
		image, _, err := ctx.Request.FormFile("image")
		if err != nil {
			ctx.String(http.StatusBadRequest, "Algo incorreto, verifique o arquivo!")
			return
		}

		defer image.Close()

		filename := filepath.Base("image_uploaded.png")
		out, err := os.Create(filename)

		if err != nil {
			ctx.String(http.StatusInternalServerError, "Failed to save file [312]")
			return
		}

		defer out.Close()

		_, err = io.Copy(out, image)
		if err != nil {
			ctx.String(http.StatusInternalServerError, "Failed to save file [231]")
			return
		}
		var Result Classification
		response, _ := http.Get("http://127.0.0.1:5000/result")
		body, err := ioutil.ReadAll(response.Body)
		if err != nil {
			fmt.Println("Erro ao ler o corpo da resposta:", err)
			return
		}
		defer response.Body.Close()

		json.Unmarshal(body, &Result)
		time.Sleep(3 * time.Second)

		ctx.HTML(http.StatusOK, "templates/result.html", gin.H{
			"message": "Arquivo enviado e processado com sucesso!",
		})
		fmt.Println("IMAGE: ", Result.Classification)
		ctx.Redirect(http.StatusSeeOther, fmt.Sprintf("/result?message=Imagem identificada como: %s", Result.Classification))
	})

	r.Run(":8080")
}
