import cv2
import mediapipe as mp
import pandas as pd
import os

class ColetorDados:
    def __init__(self):
        # Inicializa o "motor" de busca de mãos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.lista_final = []

    def coletar(self, letra):
        cap = cv2.VideoCapture(0)
        print(f"--- Coletando para a Letra: {letra} ---")
        print("Pressione 'S' para salvar o gesto ou 'Q' para sair.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1) # Espelha para facilitar o seu movimento
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = self.hands.process(img_rgb)

            if resultado.multi_hand_landmarks:
                for hand_lms in resultado.multi_hand_landmarks:
                    # Desenha os pontos na tela (seu feedback visual)
                    self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Captura a tecla
                    tecla = cv2.waitKey(1) & 0xFF
                    
                    if tecla == ord('s'):
                        coordenadas = []
                        # Extrai x e y de cada um dos 21 pontos (total 42 valores)
                        for lm in hand_lms.landmark:
                            coordenadas.extend([lm.x, lm.y])
                        
                        coordenadas.append(letra) # O "rótulo" do dado
                        self.lista_final.append(coordenadas)
                        print(f"Capturado! Total: {len(self.lista_final)}")

            cv2.imshow("TCC Libras - Coletor", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.salvar_csv()

    def salvar_csv(self):
        if self.lista_final:
            df = pd.DataFrame(self.lista_final)
            # Salva no arquivo 'dados.csv'. Se já existir, ele adiciona no final (append)
            df.to_csv('data/dados.csv', mode='a', index=False, header=not os.path.exists('data/dados.csv'))
            print("Dados salvos no CSV com sucesso!")

# Bloco principal para rodar o script
if __name__ == "__main__":
    app = ColetorDados()
    # Mude a letra aqui sempre que for coletar uma nova
    app.coletar(letra="D")