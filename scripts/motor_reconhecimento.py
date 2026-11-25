"""
motor_reconhecimento.py

Nucleo de reconhecimento de sinais: MediaPipe Hands + normalizacao das
coordenadas (+ atributos geometricos/z) + modelo treinado + estabilizacao temporal (buffer de
votacao) + filtro de confianca minima.

Este modulo NAO desenha janela nem sabe se esta rodando dentro do OpenCV
(main.py) ou de uma interface CustomTkinter (gui_app.py) - ele so recebe
um frame de imagem e devolve a letra prevista, a confianca e (quando for
o caso) a letra que acabou de se tornar "estavel". Isso evita duplicar a
logica de reconhecimento entre as duas versoes da aplicacao e garante que
elas se comportem exatamente da mesma forma.
"""

import collections
import pickle

import cv2
import mediapipe as mp

from normalizacao import construir_features

CONFIANCA_MINIMA_PADRAO = 0.65      # so aceita a previsao se a probabilidade for >= 65%
TAMANHO_BUFFER_PADRAO = 7           # quantos quadros recentes olhamos para decidir estabilidade
VOTOS_MINIMOS_PADRAO = 5            # quantos desses quadros precisam concordar


class MotorReconhecimento:
    def __init__(self, model_path='models/modelo_libras.pkl',
                 confianca_minima=CONFIANCA_MINIMA_PADRAO,
                 tamanho_buffer=TAMANHO_BUFFER_PADRAO,
                 votos_minimos=VOTOS_MINIMOS_PADRAO):
        self.model = None
        self.usa_geom = False
        self.usa_z = False
        self.especialistas = {}
        self.carregar_modelo(model_path)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.confianca_minima = confianca_minima
        self.tamanho_buffer = tamanho_buffer
        self.votos_minimos = votos_minimos
        self.buffer_predicoes = collections.deque(maxlen=tamanho_buffer)
        self.ultima_letra_estavel = ""
        self.desenhar_landmarks = True  # pode ser desligado na aba Ajustes da GUI

    def carregar_modelo(self, model_path):
        """Carrega (ou troca) o modelo usado. Aceita dois formatos:
          - formato novo (dict): {"model", "usa_geom", "usa_z",
            "especialistas"} - gerado pelo treinamento.py atual;
          - formato antigo: o proprio RandomForest salvo com pickle
            (42 coordenadas normalizadas, sem atributos extras).
        Usado pela GUI para alternar entre o modelo de Letras e o de
        Numeros sem recriar o MediaPipe."""
        with open(model_path, 'rb') as f:
            conteudo = pickle.load(f)

        if isinstance(conteudo, dict) and "model" in conteudo:
            self.model = conteudo["model"]
            self.usa_geom = bool(conteudo.get("usa_geom", False))
            self.usa_z = bool(conteudo.get("usa_z", False))
            self.especialistas = dict(conteudo.get("especialistas", {}))
        else:
            self.model = conteudo
            self.usa_geom = False
            self.usa_z = False
            self.especialistas = {}

        # O modelo e treinado com n_jobs=-1 (usa todos os nucleos no treino),
        # mas no reconhecimento em tempo real prevemos UMA amostra por quadro:
        # com n_jobs=-1 o scikit-learn abre/sincroniza varias threads a cada
        # previsao e isso custa mais do que a propria previsao (era a principal
        # causa do video "em camera lenta"). Com n_jobs=1 a previsao fica
        # varias vezes mais rapida e o resultado e identico.
        for estimador in [self.model, *self.especialistas.values()]:
            if hasattr(estimador, "n_jobs"):
                estimador.n_jobs = 1

        if hasattr(self, "buffer_predicoes"):
            self.resetar_buffer()

    # --- ajustes em tempo real (usados pelos sliders/switches da GUI) ---

    def definir_confianca_minima(self, valor):
        self.confianca_minima = float(valor)

    def definir_tamanho_buffer(self, valor):
        valor = max(1, int(round(valor)))
        self.tamanho_buffer = valor
        self.buffer_predicoes = collections.deque(self.buffer_predicoes, maxlen=valor)
        # mantem a exigencia de concordancia em ~70% do buffer, com minimo de 1
        self.votos_minimos = max(1, int(round(valor * 0.7)))

    def resetar_buffer(self):
        self.buffer_predicoes.clear()
        self.ultima_letra_estavel = ""

    # --- previsao ---

    def _montar_vetor(self, coords_brutas, z_brutos):
        """Mesmo vetor de atributos usado no treino (ver
        normalizacao.construir_features)."""
        z = z_brutos if self.usa_z else None
        return construir_features(coords_brutas, z, usar_geom=self.usa_geom)

    def _probabilidades(self, vetor):
        return self.model.predict_proba([vetor])[0]

    def _top_n(self, probs, n=3):
        indices_ordenados = probs.argsort()[::-1][:n]
        return [(self.model.classes_[i], float(probs[i])) for i in indices_ordenados]

    def _refinar_com_especialista(self, probs, top, coords_brutas, z_brutos):
        """Segunda etapa para pares de sinais parecidos (ex.: F e T): se o
        1o colocado pertence a um grupo com especialista treinado, o
        especialista (que usa a profundidade z) decide DENTRO do grupo. A
        massa de probabilidade que o modelo principal deu ao grupo inteiro
        e redistribuida conforme o especialista. Sem especialista ou sem
        z, devolve o top original. (O treinamento so mantem um especialista
        se ele nao for pior do que o mesmo par sem z - ver treinamento.py.)"""
        if not self.especialistas or z_brutos is None or not top:
            return top

        primeiro = top[0][0]
        classes = list(self.model.classes_)
        for grupo, especialista in self.especialistas.items():
            if primeiro not in grupo:
                continue

            vetor_z = construir_features(coords_brutas, z_brutos, usar_geom=True)
            p_esp = especialista.predict_proba([vetor_z])[0]
            massa = sum(float(probs[classes.index(c)]) for c in grupo if c in classes)
            novos = {classe: float(massa * p) for classe, p in zip(especialista.classes_, p_esp)}
            restantes = [(c, p) for c, p in top if c not in grupo]
            refinado = sorted(list(novos.items()) + restantes, key=lambda x: x[1], reverse=True)
            return refinado[:len(top)]
        return top

    def _prever_com_confianca(self, coords_normalizadas):
        """Mantido por compatibilidade: recebe o vetor de atributos ja
        montado e devolve (classe, confianca)."""
        probs = self._probabilidades(coords_normalizadas)
        indice_max = probs.argmax()
        return self.model.classes_[indice_max], float(probs[indice_max])

    def _letra_mais_votada_estavel(self):
        if len(self.buffer_predicoes) < self.tamanho_buffer:
            return None
        contagem = collections.Counter(self.buffer_predicoes)
        letra_mais_comum, votos = contagem.most_common(1)[0]
        if votos >= self.votos_minimos:
            return letra_mais_comum
        return None

    def processar_frame(self, frame_bgr, desenhar_landmarks=None):
        """
        Recebe um frame BGR (idealmente ja espelhado) e devolve um dict:
          - frame: o mesmo frame, com os landmarks desenhados por cima
            (se desenhar_landmarks=True) - a funcao MODIFICA o frame
            recebido, nao faz copia, pelo mesmo motivo de performance do
            codigo original;
          - mao_detectada: bool;
          - letra_atual: letra prevista neste quadro especifico (ou None
            se nenhuma mao foi detectada);
          - confianca_atual: confianca da previsao deste quadro (0 a 1);
          - letra_nova_estavel: letra que ACABOU de se tornar estavel
            neste quadro (dispara uma unica vez, quando muda em relacao
            a ultima letra estavel) - e o sinal para falar/anexar ao
            texto. None na maioria dos quadros.
        """
        if desenhar_landmarks is None:
            desenhar_landmarks = self.desenhar_landmarks
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False  # evita copia interna no MediaPipe
        resultado = self.hands.process(img_rgb)

        info = {
            "frame": frame_bgr,
            "mao_detectada": False,
            "letra_atual": None,
            "confianca_atual": 0.0,
            "letra_nova_estavel": None,
            "top3": [],
            "landmarks_brutos": None,  # 42 coordenadas (x,y) cruas, sem normalizar - usado pela coleta de dados na GUI e pelo pipeline dinamico
            "landmarks_z": None,       # 21 valores z crus (profundidade relativa ao pulso) - usado pela coleta de dados na GUI
        }

        if not resultado.multi_hand_landmarks:
            # sem mao no quadro: limpa o buffer para nao "arrastar" a
            # ultima letra detectada quando a mao sai de cena. Tambem zera
            # a "ultima letra estavel": assim, se a mao sumir e a MESMA
            # letra for mostrada de novo, ela conta como nova (permite
            # repetir letras seguidas, ex.: "LL" em "ELLA")
            self.buffer_predicoes.clear()
            self.ultima_letra_estavel = ""
            return info

        for hand_lms in resultado.multi_hand_landmarks:
            if desenhar_landmarks:
                self.mp_drawing.draw_landmarks(frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS)

            coords_brutas = []
            z_brutos = []
            for lm in hand_lms.landmark:
                coords_brutas.extend([lm.x, lm.y])
                z_brutos.append(lm.z)

            vetor = self._montar_vetor(coords_brutas, z_brutos)
            probs = self._probabilidades(vetor)
            top3 = self._top_n(probs, n=3)
            top3 = self._refinar_com_especialista(probs, top3, coords_brutas, z_brutos)
            letra_prevista, confianca = top3[0]

            info["mao_detectada"] = True
            info["letra_atual"] = letra_prevista
            info["confianca_atual"] = confianca
            info["top3"] = top3
            info["landmarks_brutos"] = coords_brutas
            info["landmarks_z"] = z_brutos

            if confianca >= self.confianca_minima:
                self.buffer_predicoes.append(letra_prevista)
            # previsao pouco confiavel: nao entra no buffer, mas tambem
            # nao reseta (evita "piscar" por causa de um unico quadro ruim)

            letra_estavel = self._letra_mais_votada_estavel()
            if letra_estavel is not None and letra_estavel != self.ultima_letra_estavel:
                info["letra_nova_estavel"] = letra_estavel
                self.ultima_letra_estavel = letra_estavel

            break  # max_num_hands=1: so processamos a primeira mao

        return info

    def fechar(self):
        self.hands.close()
