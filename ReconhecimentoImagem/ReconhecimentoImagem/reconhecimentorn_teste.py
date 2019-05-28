import os                  #Recursos do sistema operacional
import glob                #Biblioteca responsavel por pecorrer os arquivos de imagem
import dlib                #Importação das Bibliotecas dlib
import cv2                 #Importação das Bibliotecas Opencv
import numpy as np         #Importação das Bibliotecas numpy

# Criação do detector de faces
detectorFace = dlib.get_frontal_face_detector()
# Criação do detector de Pontos
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
#Esse metodo é responsavel por fazer aquela aplicação das convolucões para escolher quais são as melhores caracteristicas da imagem da face
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices.pickle")
descritoresFaciais = np.load("recursos/descritores.npy")
#variavel padrão
limiar = 0.5

#Percorrer o Diretorio fotos em JPG
for arquivo in glob.glob(os.path.join("fotos", "*.jpg")):
    imagem = cv2.imread(arquivo)
    #Mostra o retangulo na face e sua imagem com o paraetro na escala 2
    facesDetectadas = detectorFace(imagem, 2)

    #Percorre as Faces Detectadas cada um dos retangulos
    for face in facesDetectadas:
        #detecta os pontos facias
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        #Transforma no formato de lista
        listaDescritorFacial = [fd for fd in descritorFacial]
        # Transforma no formato numpy
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        #Faz o calculo da distancia euclidiana(é a distância entre dois pontos)
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        print("Distâncias: {}".format(distancias))
        #Argumento minimo da distancia das fotos
        minimo = np.argmin(distancias)
        print(minimo)
        distanciaMinima = distancias[minimo]
        print(distanciaMinima)

        #Quando o limiar for menor que 0.5 ele vai buscar na base de dados do treinamento
        if distanciaMinima <= limiar:
            #Aparece o nome junto do retangulo
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            # se nao for igual a imagens do treinamento escreve vazio
            nome = ' '

        # Mostra o retangulo nas faces
        cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
        texto = "{} {:.4f}".format(nome, distanciaMinima)
        # O texto que fica perto do retangulo com a Fonte e a COR
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

    # Mostra o nome da janela que aparece na tela
    cv2.imshow("Detector hog", imagem)
    # Pecorre uma imagem clicando no teclado
    cv2.waitKey(0)
#Destroi as janelas
cv2.destroyAllWindows()