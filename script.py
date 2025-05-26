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

diretorio = "lfw_funneled"
nome_colecao = "lfw_faces"
tamanho_vetor = 128     # Número de dimensões do embedding do face_recognition

# Conecta ao Qdrant
client = QdrantClient("http://localhost:6333")

# Cria a coleção se não existir
if nome_colecao not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=nome_colecao,
        vectors_config=VectorParams(size=tamanho_vetor, distance=Distance.COSINE),
    )

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

def processa_imagens(dir, step=1):
    '''
    Função usada para processar todos os diretórios e fotos dentro de lfw_funneled.

    Args:
        dir: Nome do diretório que contém as fotos.
        step: Parâmetro para indicar o intervalo de diretórios processados. Utilizado caso não seja necessário processar todos os diretórios contidos, economizando tempo de processamento.
    
    Returns:
        return: Vetor com as imagens processas em formato de PointStruct para serem inseridas na coleção.
    '''
    pontos = []
    todos_diretorios = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    diretorios_selecionados = todos_diretorios[::step]      # Seleciona os diretórios com o intervalo determinado

    for pessoa in tqdm(diretorios_selecionados, desc=f"Processando 1 a cada {step} pessoas"):
        dir_pessoa = os.path.join(dir, pessoa)

        for nome_arq in os.listdir(dir_pessoa):
            if not nome_arq.endswith(".jpg"):
                continue

            caminho_imagem = os.path.join(dir_pessoa, nome_arq)
            imagem = face_recognition.load_image_file(caminho_imagem)
            embeddings = face_recognition.face_encodings(imagem)

            if len(embeddings) == 0:
                continue    # Pula se não encontrar rostos

            embedding_pessoa = embeddings[0]    # Presume que o primeiro e único rosto encontrado será da pessoa nomeada

            id_ponto = str(uuid.uuid4())    # ID único 

            # Cria estrutura de ponto para inserir na coleção
            pontos.append(
                PointStruct(
                    id=id_ponto,
                    vector=embedding_pessoa.tolist(),
                    payload={"nome": pessoa, "arquivo": nome_arq}
                )
            )
        
        return pontos