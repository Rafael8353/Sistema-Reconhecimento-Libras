"""
audio_async.py

Fala assincrona (Text-to-Speech) usando pyttsx3, extraida do main.py para
ser reutilizada tanto pela versao simples (janela OpenCV) quanto pela
interface grafica (gui_app.py).

Historico do bug "fala a primeira letra e depois emudece" (Windows):
  1a tentativa: manter UMA thread de audio viva o tempo todo, com UM
    engine criado dentro dela (resolveu o problema de thread errada criar
    o engine, mas o emudecimento depois da 1a fala continuou).
  2a tentativa: manter a mesma thread, mas criar um engine NOVO
    (pyttsx3.init()) a cada fala. Nao resolveu completamente: na pratica,
    o pyttsx3 guarda os engines num cache interno (por nome de driver), e
    mesmo "descartando" a variavel Python, o objeto COM antigo as vezes
    nao e liberado a tempo pelo coletor de lixo, entao o pyttsx3.init()
    seguinte podia devolver o MESMO engine (ja "preso") por baixo dos
    panos.
  3a tentativa (atual): alem de um engine novo, cada fala roda numa
    THREAD do sistema operacional totalmente nova e descartavel - a
    thread e criada, fala UMA frase, e termina. Isso forca o Windows a
    encerrar de verdade o apartamento COM daquela fala quando a thread
    morre, em vez de depender do coletor de lixo do Python. A fila
    (queue.Queue) + uma thread "despachante" fixa continuam garantindo
    que as falas saem na ordem certa e nunca duas ao mesmo tempo (o
    despachante so pega a proxima fala da fila depois que a thread da
    fala anterior realmente terminou - thread.join()).
"""

import queue
import threading

import pyttsx3


class FalanteAssincrono:
    def __init__(self):
        self._fila = queue.Queue()
        self._thread_despachante = threading.Thread(target=self._despachante, daemon=True)
        self._thread_despachante.start()

    def _despachante(self):
        """Roda pra sempre numa unica thread fixa, so para tirar textos da
        fila IN ORDEM e, para cada um, criar e aguardar uma thread nova e
        descartavel que efetivamente fala (ver explicacao no topo do
        arquivo)."""
        while True:
            texto = self._fila.get()
            if texto is None:  # sinal de encerramento
                break
            if not texto:
                continue

            thread_desta_fala = threading.Thread(target=self._falar_uma_vez, args=(texto,))
            thread_desta_fala.start()
            thread_desta_fala.join()  # espera esta fala terminar antes de pegar a proxima

    def _falar_uma_vez(self, texto):
        try:
            engine = pyttsx3.init()
            engine.say(texto)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"Aviso: falha ao tentar falar '{texto}' ({e}).")

    def falar(self, texto):
        self._fila.put(texto)

    def parar(self):
        self._fila.put(None)
