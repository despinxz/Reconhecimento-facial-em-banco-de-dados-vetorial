import os
import uuid
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import face_recognition
import face_recognition_models
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import Filter, SearchRequest
import matplotlib.pyplot as plt
from PIL import Image

diretorio = "lfw_funneled_reduzido"
nome_colecao = "lfw_faces"
tamanho_vetor = 128     # Número de dimensões do embedding do face_recognition

def cria_conexao():
    # Conecta ao Qdrant
    client = QdrantClient("http://localhost:6333")

    # Cria a coleção se não existir
    if nome_colecao not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=nome_colecao,
            vectors_config=VectorParams(size=tamanho_vetor, distance=Distance.COSINE),
        )

    return client

def input_imagem():
    '''
    Função usada para permitir input de imagens pelo explorador de arquivos.
    
    Returns:
        return: Caminho do arquivo selecionado.
    '''
    root = tk.Tk()
    root.withdraw() 

    file_name = filedialog.askopenfilename(
        title="Selecione um arquivo de imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos os arquivos", "*.*")]
    )
    
    return file_name

def processa_imagem(caminho_imagem):
    '''
    Função usada para processar uma imagem passada como parâmetro.

    Args:
        caminho_imagem: Caminho da imagem passada como parâmetro.
    
    Returns:
        return: Vetor com as imagens processas em formato de PointStruct para serem inseridas na coleção.
    '''
    nome = caminho_imagem.split("\\\\")[-1].split('.')[0]
    if not caminho_imagem.endswith(".jpg"):
        return
    
    imagem = face_recognition.load_image_file(caminho_imagem)
    embeddings = face_recognition.face_encodings(imagem)

    if len(embeddings) == 0:
        return    # Retorna se não encontrar rostos

    embedding_pessoa = embeddings[0]    # Presume que o primeiro e único rosto encontrado será da pessoa nomeada

    id_ponto = str(uuid.uuid4())    # ID único 

    # Cria estrutura de ponto para inserir na coleção
    ponto = PointStruct(
                id=id_ponto,
                vector=embedding_pessoa.tolist(),
                payload={"nome": nome, "arquivo": caminho_imagem}
            )

    return ponto
    
def insere_imagem_colecao(client, vetor_ponto_imagem):
    '''
    Função usada para inserir uma imagem na coleção do QDrant.

    Args:
        client: Conexão com QDrant.
        imagem: Vetor de pontos da imagem.
    '''
    client.upsert(collection_name="lfw_faces", points=vetor_ponto_imagem)
    return

def busca_imagens_semelhantes(client, vetor_ponto_imagem, top_k=5):
    '''
    Função usada para buscar as K imagens mais semelhantes na coleção.

    Args:
        client: Conexão com QDrant.
        imagem: Caminho da imagem a ser buscada.
        top_k: Quantidade de fotos semelhantes a serem buscadas.

    Returns:
        return: Vetor com os dados das 5 imagens mais semelhantes.
    '''
    resultado = client.search(collection_name="lfw_faces_reduzido", 
                              query_vector=vetor_ponto_imagem, 
                              limit=top_k)
    
    return resultado

def exibe_resultados(semelhantes):
    '''
    Exibe imagens semelhantes com nome e score.

    Args:
        semelhantes: Lista de resultados da busca no QDrant.
    '''
    num_resultados = len(semelhantes)
    plt.figure(figsize=(15, 5))

    for i, item in enumerate(semelhantes):
        payload = item.payload
        nome = payload.get("nome", "Desconhecido")
        arquivo = payload.get("arquivo")
        score = item.score

        try:
            imagem = Image.open(arquivo)
            plt.subplot(1, num_resultados, i + 1)
            plt.imshow(imagem)
            plt.axis("off")
            plt.title(f"{nome}\nScore: {score:.2f}")
        except Exception as e:
            print(f"Erro ao carregar {arquivo}: {e}")

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    client = cria_conexao()
    imagem = input_imagem()
    vetor_ponto_imagem = processa_imagem(imagem)
    semelhantes = busca_imagens_semelhantes(client, vetor_ponto_imagem.vector, top_k=5)
    exibe_resultados(semelhantes)
    # insere_imagem_colecao(vetor_ponto_imagem)