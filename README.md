# Projeto de reconhecimento facial com banco de dados vetorial

## Integrantes
- Ana Clara das Neves Barreto
- Eloisa Antero Guisse
- Jamyle Gonçalves Rodrigues Silva
- Sarah Klock Mauricio

## Sobre o projeto 
Este repositório contém o notebook e o script desenvolvidos para a implementação de um banco de dados vetorial para reconhecimento facial. O projeto foi desenvolvido como parte do curso de Tópicos Especiais em Bancos de Dados na Universidade de São Paulo (USP), com o objetivo de explorar o funcionamento de bancos de dados não-relacionais.

## Estrutura do repositório
- **main.ipynb**: Métodos para testes com o banco de dados;
- **script.py**: Script principal, que ao ser executado, inicializa a coleção no QDrant, e pede para o usuário passar uma imagem para a busca de rostos semelhantes.

## Como usar
Para utilizar o sistema e o script, é necessário: 

1. **Ter o Docker instalado**
Antes de instalar o Docker, certifique-se de ter o WSL2 instalado. Execute no PowerShell:

```
wsl --install
```

Instale o Docker a partir do seguinte link: https://www.docker.com/products/docker-desktop/

2. **Instale as dependências**
Execute no terminal:

```
pip install -r requirements.txt
```

3. **Inicie o Docker**
Execute no PowerShell:

```
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. **Insira o diretório lfw_funneled no diretório do projeto**
A base está disponível em: https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data 

Após todas as configurações serem feitas, tanto o script principal quanto os códigos no notebook podem ser executados.