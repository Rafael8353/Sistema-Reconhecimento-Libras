"""
main.py

Versao simples (janela do OpenCV) do tradutor de Libras em tempo real.
Util para testes rapidos sem abrir a interface grafica completa.

Para a versao com interface grafica (CustomTkinter, com area de texto
acumulando letras em palavras, botoes de iniciar/pausar, seletores de
configuracao, etc - o "próximo passo" descrito no TCC1), veja
scripts/gui_app.py.

A logica de reconhecimento (MediaPipe + normalizacao + modelo + buffer
de estabilidade) vive em motor_reconhecimento.py e e compartilhada pelas
duas versoes, para nao haver duas implementacoes que podem divergir.
"""

import cv2

from audio_async import FalanteAssincrono
from motor_reconhecimento import MotorReconhecimento


class TradutorLibras:
    def __init__(self, model_path='models/modelo_libras.pkl'):
        self.motor = MotorReconhecimento(model_path=model_path)
        self.falante = FalanteAssincrono()

    def iniciar(self):
        cap = cv2.VideoCapture(0)
        print("Tradutor Iniciado! Posicione a mao na camera. Pressione 'Q' para sair.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            info = self.motor.processar_frame(frame)

            if info["letra_atual"]:
                confiante = info["confianca_atual"] >= self.motor.confianca_minima
                cor = (0, 255, 0) if confiante else (0, 165, 255)
                cv2.putText(frame, f"Letra: {info['letra_atual']} ({info['confianca_atual']*100:.0f}%)",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, cor, 3)
            cv2.putText(frame, f"Falada: {self.motor.ultima_letra_estavel}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Top-3 previsoes (util para diagnosticar letras que nao sao
            # bem reconhecidas: mostra se a letra certa aparece em 2o/3o
            # lugar, o que aponta para confusao com outra letra parecida)
            if info["top3"]:
                texto_top3 = "  ".join(f"{letra}:{conf*100:.0f}%" for letra, conf in info["top3"])
                cv2.putText(frame, texto_top3, (50, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if info["letra_nova_estavel"]:
                self.falante.falar(info["letra_nova_estavel"])

            cv2.imshow("TCC Libras - TRADUTOR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.falante.parar()
        self.motor.fechar()


if __name__ == "__main__":
    app = TradutorLibras()
    app.iniciar()
