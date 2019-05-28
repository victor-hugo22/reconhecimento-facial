import cv2   # Importação das Bibliotecas Opencv
import dlib  # Importação das Bibliotecas dlib

#Fonte da Letra do classificador
fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

#Carregamento das Imagens
imagem = cv2.imread("fotos/grupo.0.jpg")
#imagem = cv2.imread("fotos/grupo.1.jpg")
#imagem = cv2.imread("fotos/grupo.2.jpg")
#imagem = cv2.imread("fotos/grupo.3.jpg")
#imagem = cv2.imread("fotos/grupo.4.jpg")
#imagem = cv2.imread("fotos/grupo.5.jpg")
#imagem = cv2.imread("fotos/grupo.6.jpg")
#imagem = cv2.imread("fotos/grupo.7.jpg")

# Haar
detectorHaar = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
#Converter essa iagem para escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#Onde fica os quadrados nas faces
facesDetectadasHaar = detectorHaar.detectMultiScale(imagemCinza, scaleFactor=1.1, minSize=(10,10))

# Hog
detectorHog = dlib.get_frontal_face_detector()
#Aparece os retangulos, a pontuação(valor da confiança),vai indicar qual a posição de faces.Metodo run, retorna os valores adicionais da confiança
facesDetectadasHog, pontuacao, idx = detectorHog.run(imagem, 2)

#Percorrer os retangulos do HOG e mostrar a pontuação de cada face
for i, d in enumerate(facesDetectadasHog):
    print(pontuacao[i])
print("HAARCASCADE")
#Percorrer os retangulos do HAARCASCADE. e mostrar a pontuação de cada face
for i, d in enumerate(facesDetectadasHaar):
    print(pontuacao[i])


# Percorre todas as Faces detectadas na imagem usando o descritor HAARCASCADE
for (x, y, l, a) in facesDetectadasHaar:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
    cv2.putText(imagem, "Haar", (x, y - 5), fonte, 0.5, (0, 255, 0))

# Percorre todas as Faces detectadas na imagem usando o descritor HOG
for face in facesDetectadasHog:
    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    #Mostra o retangulo nas Faces
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
    #O texto que fica perto do retangulo
    cv2.putText(imagem, "Hog", (d, t), fonte, 0.5, (0, 255, 255))

# Mostra o nome da janela que aparece na tela
cv2.imshow("Comparativo detectores", imagem)
cv2.waitKey(0)
#Destroi as janelas
cv2.destroyAllWindows()