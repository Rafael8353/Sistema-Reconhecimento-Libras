import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading

class TradutorLibras:
    def __init__(self, model_path='models/modelo_libras.pkl'):
        # 1. Configurar Áudio (Text-to-Speech)
        self.engine = pyttsx3.init()
        
        # 2. Carregar o Cérebro (Modelo)
        with open(model_path, 'rb') as f:
            self.model = pickle.dump = pickle.load(f)

        # 3. Configurar Visão
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.ultima_letra = ""

    def falar(self, texto):
        """Função para falar sem travar o vídeo (usando Threads)"""
        def thread_fala():
            self.engine.say(texto)
            self.engine.runAndWait()
        threading.Thread(target=thread_fala).start()

    def iniciar(self):
        cap = cv2.VideoCapture(0)
        print("Tradutor Iniciado! Posicione a mão na câmera.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = self.hands.process(img_rgb)

            if resultado.multi_hand_landmarks:
                for hand_lms in resultado.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extrair coordenadas para a predição
                    coords = []
                    for lm in hand_lms.landmark:
                        coords.extend([lm.x, lm.y])
                    
                    # 4. A IA tenta adivinhar a letra
                    predicao = self.model.predict([coords])
                    letra_detectada = predicao[0]

                    # Exibir na tela
                    cv2.putText(frame, f"Letra: {letra_detectada}", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    # 5. Se a letra mudou, o sistema fala
                    if letra_detectada != self.ultima_letra:
                        self.falar(letra_detectada)
                        self.ultima_letra = letra_detectada

            cv2.imshow("TCC Libras - TRADUTOR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = TradutorLibras()
    app.iniciar()