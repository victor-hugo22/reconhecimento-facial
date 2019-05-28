import os                    #Recursos do sistema operacional
import glob                  #Biblioteca responsavel por pecorrer os arquivos de imagem
import _pickle as cPickle    #utilizamos para gravação dos arquivos de Treinamneto
import dlib                  #Importação das Bibliotecas dlib
import cv2                   #Importação das Bibliotecas Opencv
import numpy as np           #Importação das Bibliotecas numpy

# Detectar a imagem
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
#Variavel para fazer efetivamente o reconhecimento facial
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

#Pecorre os arquivos das imagens que estão na pasta Treinamento
for arquivo in glob.glob(os.path.join("fotos/treinamento", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 1) # Aumenta a escala da imagem
    numeroFacesDetectadas = len(facesDetectadas) # Mostra a quantidade de faces detectadas
    print(numeroFacesDetectadas)

    #Validação ds imagem
    if numeroFacesDetectadas > 1:
        print("Há mais de uma face na imagem {}".format(arquivo))
        exit(0)
    elif numeroFacesDetectadas < 1:
        print("Nenhuma face encontrada no arquivo {}".format(arquivo))
        exit(0)

    #vai Detectar 128 caracteristicas facias
    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        #Lista dos Descritores Faciais
        listaDescritorFacial = [df for df in descritorFacial]
        #print(listaDescritorFacial)

        #Converção para o formato numpy para poder gerar os arquivos
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        #print(npArrayDescritorFacial)

        #Criação de mas uma coluna
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        #print(npArrayDescritorFacial)

        #Recebe a variavel npArrayDescritorFacial se ele for vazio
        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)
        #Armazenar o numero do arquivo no IDX
        indice[idx] = arquivo
        idx += 1

    #TESTAR A IMAGEM
    cv2.imshow("Treinamento", imagem) # Mostra a imagem na tela
    cv2.waitKey(0)  #Pecorre uma imagem clicando no teclado

# Mostra o Tamanho e o Formato da Matriz do if descritoresFaciais
print("Tamanho: {} Formato: {}".format(len(descritoresFaciais), descritoresFaciais.shape))
print(descritoresFaciais)
print(indice)

# GERA OS ARQUIVOS PARA PASTA RECURSOS
np.save("recursos/descritores.npy", descritoresFaciais)
with open("recursos/indices.pickle", 'wb') as f:
    cPickle.dump(indice, f)

#Destroi as janelas
cv2.destroyAllWindows()