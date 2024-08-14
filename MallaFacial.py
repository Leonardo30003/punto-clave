import cv2
import mediapipe as mp
import math

#--------------------------Realizando la Videocaptura------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Definimos el ancho de la ventana
cap.set(4, 720) # Definimos el alto de la ventana

#-----------------------Creamos nuestra función de dibujo-------------
mp_dibujo = mp.solutions.drawing_utils
config_dibujo = mp_dibujo.DrawingSpec(thickness=1, circle_radius=1) # Ajustamos la configuración del dibujo

#--------Creamos un objeto donde almacenaremos la malla facial----------
mp_malla_facial = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#--------Creamos el while principal-------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #-------Corrección de color-----------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #------Observamos los resultados-------------
    resultados = mp_malla_facial.process(frame_rgb)

    # Creamos unas listas donde almacenaremos los resultados
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:  # Si detectamos algún rostro
        for rostro in resultados.multi_face_landmarks:  # Mostramos el rostro detectado
            mp_dibujo.draw_landmarks(
                frame, 
                rostro, 
                mp.solutions.face_mesh.FACEMESH_TESSELATION, 
                config_dibujo, 
                config_dibujo
            )

            # Ahora vamos a extraer los puntos del rostro detectado
            for id, punto in enumerate(rostro.landmark):
                alto, ancho, _ = frame.shape
                x, y = int(punto.x * ancho), int(punto.y * alto)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                
                if len(lista) == 468:
                    # Ceja Derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    # Ceja Izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # Boca Extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)

                    # Boca Aperturas
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                    # Clasificación de emociones
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona enojada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 20:
                        cv2.putText(frame, 'Persona Feliz', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud4 < 90 and longitud4 > 20:
                        cv2.putText(frame, 'Persona Asombrada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 5:
                        cv2.putText(frame, 'Persona Triste', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Reconocimiento de Emociones', frame)
    t = cv2.waitKey(1)
    if t == 27:  # 27 es el código ASCII para la tecla 'ESC'
        break

cap.release()
cv2.destroyAllWindows()
