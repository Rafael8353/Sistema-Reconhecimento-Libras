"""
gui_app.py

Interface grafica (CustomTkinter) do Tradutor de Libras.

Organizacao da tela
-------------------
Barra lateral com quatro paginas e o botao global da camera:
  - Traduzir : video, letra reconhecida, texto acumulado e voz;
  - Coletar  : grava amostras de letras, numeros e sinais com movimento;
  - Treinar  : treina/avalia os modelos e apaga amostras;
  - Ajustes  : camera, sensibilidade do reconhecimento e "sobre".

Por que a camera roda em threads separadas
------------------------------------------
Na versao anterior, a leitura da webcam, o MediaPipe, o modelo e o desenho
do video aconteciam todos na thread da interface. Quando o processamento de
um quadro demorava mais que o intervalo da camera, os quadros se acumulavam
no buffer do driver e o video ia ficando atrasado ("camera lenta"). Agora:
  * uma thread de CAPTURA le a webcam sem parar e guarda so o quadro mais
    recente (os antigos sao descartados, entao o atraso nao se acumula);
  * uma thread de PROCESSAMENTO pega sempre o quadro mais recente, roda o
    MediaPipe + modelo e publica o resultado;
  * a interface so desenha o ultimo resultado e trata os eventos (letra
    estavel, gravacao de sinal com movimento).
Pausar encerra as duas threads e libera a camera; Ligar cria threads novas.

Como executar:
    python scripts/gui_app.py

Dependencias adicionais em relacao ao resto do projeto:
    pip install customtkinter pillow
"""

import collections
import contextlib
import io
import pickle
import queue
import re
import sys
import threading
import time
import tkinter as tk
import tkinter.messagebox as messagebox
from pathlib import Path

import cv2

try:
    import customtkinter as ctk
except ImportError:
    raise SystemExit(
        "A biblioteca 'customtkinter' nao esta instalada.\n"
        "Instale com: pip install customtkinter pillow"
    )

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit(
        "A biblioteca 'Pillow' nao esta instalada.\n"
        "Instale com: pip install customtkinter pillow"
    )

import pandas as pd

from audio_async import FalanteAssincrono
from motor_reconhecimento import (
    CONFIANCA_MINIMA_PADRAO,
    TAMANHO_BUFFER_PADRAO,
    MotorReconhecimento,
)
from coletor import remover_ultima_amostra, salvar_amostra
from contribuicoes import exportar_contribuicao
from modos import MODOS, NUMEROS, modo_do_rotulo
from remover_amostras import remover_amostras as remover_amostras_dataset
from treinamento import TreinadorIA
from avaliacao import main as avaliar_modelos_main
from dinamico import (
    LETRAS_DINAMICAS,
    NUM_QUADROS_JANELA,
    NUM_QUADROS_PREPARO,
    achatar_sequencia_bruta,
    normalizar_sequencia,
    remover_ultima_amostra_dinamica,
    salvar_amostra_dinamica,
)
from treinamento_dinamico import TreinadorDinamicoIA

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = MODOS["letras"]["modelo"]
CAMINHO_DATASET = MODOS["letras"]["csv"]
CAMINHO_MODELO_DINAMICO = RAIZ / "models" / "modelo_libras_dinamico.pkl"
CAMINHO_DATASET_DINAMICO = RAIZ / "data" / "dados_dinamicos.csv"

INTERVALO_UI_MS = 15  # frequencia com que a interface busca novos quadros processados

LETRAS_ALFABETO = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LETRAS_ESTATICAS = [c for c in LETRAS_ALFABETO if c not in LETRAS_DINAMICAS]  # A-Z sem J, X, Z

# Tipos de sinal na pagina Coletar (sem repeticoes entre eles)
TIPOS_COLETA = {
    "Letras": LETRAS_ESTATICAS,
    "Movimento": LETRAS_DINAMICAS,   # J, X, Z, Ç
    "Números": NUMEROS,
}
# Na remocao, "Letras" mostra o A-Z completo: podem existir capturas
# ESTATICAS antigas de J, X e Z em data/dados.csv.
TIPOS_REMOVER = {
    "Letras": LETRAS_ALFABETO,
    "Números": NUMEROS,
    "Movimento": LETRAS_DINAMICAS,
}
RESOLUCOES = {
    "Padrão (640×480)": (640, 480),
    "HD (1280×720)": (1280, 720),
}

# ----------------------------------------------------------------------
# Paleta (tema escuro fixo)
# ----------------------------------------------------------------------
COR_FUNDO = "#0e1116"
COR_LATERAL = "#131821"
COR_CARTAO = "#1a202b"
COR_CARTAO_2 = "#232b38"
COR_BORDA = "#2c3545"
COR_TEXTO = "#e9edf5"
COR_SUAVE = "#8b95a9"
COR_DESTAQUE = "#4f8cff"
COR_DESTAQUE_HOVER = "#3a73e0"
COR_SUCESSO = "#22c55e"
COR_SUCESSO_HOVER = "#16a34a"
COR_ALERTA = "#f59e0b"
COR_PERIGO = "#ef4444"
COR_PERIGO_HOVER = "#c93636"
COR_VIDEO = "#0a0c10"


def caminho_dataset_estatico(rotulo):
    """numeros -> dados_numeros.csv, letras -> dados.csv."""
    return MODOS[modo_do_rotulo(rotulo)]["csv"]


def resumo_da_saida(saida):
    """Extrai da saida de texto do treinamento uma linha curta com a
    acuracia mais 'honesta' disponivel (blocos de tempo > validacao
    cruzada > holdout)."""
    linhas = (saida or "").splitlines()
    for chave in ("blocos de tempo", "validacao cruzada", "acuracia"):
        for linha in linhas:
            if chave in linha.lower():
                achado = re.search(r"(\d+[.,]\d+)\s*%", linha)
                if achado:
                    return f"Acerto estimado: {achado.group(1).replace('.', ',')}%"
    return None


def abrir_camera(indice, largura, altura):
    """Abre a webcam tentando os backends mais estaveis primeiro. No
    Windows, logo depois de uma camera ser liberada, o primeiro open as
    vezes falha: por isso ha uma segunda rodada de tentativas."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for _rodada in range(2):
        for backend in backends:
            cap = cv2.VideoCapture(indice, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if largura > 640:
                # em HD, MJPG costuma manter 30 fps; o formato cru (YUY2)
                # em 1280x720 cai para 5-10 fps em muitas webcams
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(15):
                ok, _quadro = cap.read()
                if ok:
                    return cap
                time.sleep(0.05)
            cap.release()
        time.sleep(0.6)
    return None


class CameraWorker:
    """Captura + processamento fora da thread da interface (ver docstring
    do modulo). Os resultados de processar_frame() ficam em self.saida
    (deque), consumidos pela interface."""

    def __init__(self, indice, resolucao, motor, lock_motor):
        self.indice = indice
        self.resolucao = resolucao
        self.motor = motor
        self.lock_motor = lock_motor
        self.saida = collections.deque(maxlen=90)
        self.erro = None
        self.aberta = False
        self._parar = threading.Event()
        self._cond = threading.Condition()
        self._quadro = None
        self._id_quadro = 0
        self._threads = []

    @property
    def ativo(self):
        return not self._parar.is_set()

    def iniciar(self):
        self._threads = [
            threading.Thread(target=self._loop_captura, daemon=True),
            threading.Thread(target=self._loop_processamento, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def parar(self):
        self._parar.set()
        with self._cond:
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=2.0)

    def _loop_captura(self):
        cap = abrir_camera(self.indice, *self.resolucao)
        if cap is None:
            self.erro = (f"Não consegui abrir a câmera {self.indice}.\n\n"
                         "Verifique se outro programa está usando a câmera "
                         "ou escolha outra em Ajustes.")
            self._encerrar()
            return

        self.aberta = True
        falhas = 0
        try:
            while not self._parar.is_set():
                ok, quadro = cap.read()
                if not ok or quadro is None:
                    falhas += 1
                    if falhas >= 60:
                        self.erro = "A câmera parou de enviar imagens."
                        break
                    time.sleep(0.02)
                    continue
                falhas = 0
                with self._cond:
                    self._quadro = quadro
                    self._id_quadro += 1
                    self._cond.notify()
        finally:
            cap.release()
            self._encerrar()

    def _loop_processamento(self):
        visto = 0
        while True:
            with self._cond:
                while self._id_quadro == visto and not self._parar.is_set():
                    self._cond.wait(timeout=0.2)
                if self._parar.is_set():
                    return
                quadro = self._quadro
                visto = self._id_quadro

            quadro = cv2.flip(quadro, 1)
            try:
                with self.lock_motor:
                    info = self.motor.processar_frame(quadro)
            except Exception as e:  # nao deixa a thread morrer em silencio
                self.erro = f"Erro no reconhecimento: {e}"
                self._encerrar()
                return
            self.saida.append(info)

    def _encerrar(self):
        self._parar.set()
        with self._cond:
            self._cond.notify_all()


class AplicativoLibras(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("Tradutor de Libras")
        self.geometry("1220x780")
        self.minsize(1060, 700)
        self.configure(fg_color=COR_FUNDO)
        self.protocol("WM_DELETE_WINDOW", self._ao_fechar)
        if sys.platform.startswith("win"):
            self.after(0, lambda: self.state("zoomed"))  # abre maximizada no Windows

        # --- estado geral ---
        self.motor = None
        self.lock_motor = threading.Lock()
        self.camera = None  # CameraWorker ativo (ou None)
        self.modelo_dinamico = None
        self.falante = FalanteAssincrono()
        self.modo_reconhecimento = "letras"
        self._pagina_atual = None
        self._foto_video = None  # ImageTk.PhotoImage reaproveitada entre quadros
        self._label_video_ativo = None
        self._ultimo_info = None
        self._tempos_quadros = collections.deque(maxlen=30)
        self._after_id = None
        self._after_id_ferramentas = None
        self._after_id_toast = None
        self._fila_ferramentas = queue.Queue()

        # --- variaveis de interface ---
        self.var_audio_ligado = ctk.BooleanVar(value=True)
        self.var_auto_adicionar = ctk.BooleanVar(value=True)
        self.var_desenhar = ctk.BooleanVar(value=True)
        self.var_camera = ctk.StringVar(value="Câmera 0")
        self.var_resolucao = ctk.StringVar(value=list(RESOLUCOES)[0])
        self.var_modo = ctk.StringVar(value="Letras")

        # --- correcao rapida (trocar a ultima letra por uma alternativa) ---
        self._alternativas = []
        self._ultima_letra_adicionada = None

        # --- coleta ---
        self.var_tipo_coleta = ctk.StringVar(value="Letras")
        self.var_letra_coleta = ctk.StringVar(value=LETRAS_ESTATICAS[0])
        self._botoes_letras_coleta = {}
        self._contagem_sessao_coleta = 0
        self._pode_desfazer_coleta = False
        self._ultima_coleta = None  # (caminho, pessoa, letra, dinamica)
        self._cache_contagens = {}

        # --- gravacao de sinais com movimento (coleta ou reconhecimento) ---
        self._gravando_dinamico = False
        self._modo_gravacao_dinamica = None  # "coleta" ou "reconhecer"
        self._fase_gravacao_dinamica = None  # "preparo" ou "gravando"
        self._contagem_gravacao_dinamica = 0
        self._buffer_dinamico = []
        self._pessoa_gravacao_dinamica = None
        self._letra_gravacao_dinamica = None

        self._carregar_modelo()
        self._carregar_modelo_dinamico()
        self._montar_layout()
        self._mostrar_pagina("traduzir")
        self.bind_all("<space>", self._atalho_espaco)
        self._after_id = self.after(INTERVALO_UI_MS, self._loop_ui)

    # ==================================================================
    # Modelos
    # ==================================================================

    def _carregar_modelo(self):
        if not CAMINHO_MODELO.exists():
            messagebox.showerror(
                "Modelo não encontrado",
                "Ainda não existe um modelo treinado.\n\n"
                "Rode uma vez: python scripts/treinamento.py",
            )
            raise SystemExit(1)
        self.motor = MotorReconhecimento(model_path=str(CAMINHO_MODELO))

    def _carregar_modelo_dinamico(self):
        """Modelo dos sinais com movimento (opcional: o app abre sem ele)."""
        if CAMINHO_MODELO_DINAMICO.exists():
            with open(CAMINHO_MODELO_DINAMICO, "rb") as f:
                self.modelo_dinamico = pickle.load(f)
            if hasattr(self.modelo_dinamico, "n_jobs"):
                self.modelo_dinamico.n_jobs = 1
        else:
            self.modelo_dinamico = None

    # ==================================================================
    # Helpers de estilo
    # ==================================================================

    def _fonte(self, tamanho, peso="normal"):
        return ctk.CTkFont(size=tamanho, weight=peso)

    def _cartao(self, pai, **kw):
        return ctk.CTkFrame(pai, fg_color=COR_CARTAO, corner_radius=16, **kw)

    def _botao(self, pai, texto, comando, estilo="primario", **kw):
        estilos = {
            "primario": dict(fg_color=COR_DESTAQUE, hover_color=COR_DESTAQUE_HOVER, text_color="white"),
            "sucesso": dict(fg_color=COR_SUCESSO, hover_color=COR_SUCESSO_HOVER, text_color="white"),
            "perigo": dict(fg_color=COR_PERIGO, hover_color=COR_PERIGO_HOVER, text_color="white"),
            "secundario": dict(fg_color=COR_CARTAO_2, hover_color=COR_BORDA, text_color=COR_TEXTO),
            "fantasma": dict(fg_color="transparent", hover_color=COR_CARTAO_2, text_color=COR_SUAVE,
                             border_width=1, border_color=COR_BORDA),
        }
        opcoes = dict(corner_radius=10, height=40, font=self._fonte(14, "bold"))
        opcoes.update(estilos[estilo])
        opcoes.update(kw)
        return ctk.CTkButton(pai, text=texto, command=comando, **opcoes)

    def _rotulo(self, pai, texto, tamanho=14, peso="normal", cor=COR_TEXTO, **kw):
        return ctk.CTkLabel(pai, text=texto, font=self._fonte(tamanho, peso), text_color=cor, **kw)

    def _pilula(self, pai, texto=""):
        return ctk.CTkLabel(pai, text=texto, corner_radius=12, fg_color=COR_CARTAO_2,
                            text_color=COR_SUAVE, font=self._fonte(13), height=30, padx=12)

    def _definir_pilula(self, pilula, texto, cor):
        pilula.configure(text=f"●  {texto}", text_color=cor)

    def _area_video(self, pai):
        """Cartao com um tk.Label simples para o video (bem mais rapido que
        CTkImage para atualizar ~30 vezes por segundo)."""
        cartao = ctk.CTkFrame(pai, fg_color=COR_VIDEO, corner_radius=16)
        cartao.pack_propagate(False)
        cartao.grid_propagate(False)
        label = tk.Label(cartao, bg=COR_VIDEO, fg=COR_SUAVE, bd=0, highlightthickness=0,
                         text="Câmera desligada\n\nclique em  ▶ Ligar câmera  na barra lateral",
                         font=("Segoe UI", 13), justify="center")
        label.pack(fill="both", expand=True, padx=10, pady=10)
        label._cartao = cartao
        return cartao, label

    # ==================================================================
    # Layout
    # ==================================================================

    def _montar_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._montar_barra_lateral()

        self.area_conteudo = ctk.CTkFrame(self, fg_color=COR_FUNDO, corner_radius=0)
        self.area_conteudo.grid(row=0, column=1, sticky="nsew")
        self.area_conteudo.grid_rowconfigure(0, weight=1)
        self.area_conteudo.grid_columnconfigure(0, weight=1)

        self.paginas = {}
        for nome, montar in (("traduzir", self._montar_pagina_traduzir),
                             ("coletar", self._montar_pagina_coletar),
                             ("treinar", self._montar_pagina_treinar),
                             ("ajustes", self._montar_pagina_ajustes)):
            pagina = ctk.CTkFrame(self.area_conteudo, fg_color=COR_FUNDO, corner_radius=0)
            pagina.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
            montar(pagina)
            self.paginas[nome] = pagina

        self.toast = ctk.CTkLabel(self, text="", corner_radius=12, fg_color=COR_CARTAO_2,
                                  text_color=COR_TEXTO, font=self._fonte(14), height=40, padx=18)

    def _montar_barra_lateral(self):
        barra = ctk.CTkFrame(self, fg_color=COR_LATERAL, corner_radius=0, width=230)
        barra.grid(row=0, column=0, sticky="nsw")
        barra.grid_propagate(False)
        barra.grid_columnconfigure(0, weight=1)
        barra.grid_rowconfigure(6, weight=1)

        self._rotulo(barra, "Libras", 26, "bold").grid(
            row=0, column=0, sticky="w", padx=22, pady=(26, 0))
        self._rotulo(barra, "Tradutor em tempo real", 13, cor=COR_SUAVE).grid(
            row=1, column=0, sticky="w", padx=22, pady=(0, 26))

        self.botoes_nav = {}
        itens = (("traduzir", "Traduzir"), ("coletar", "Coletar"),
                 ("treinar", "Treinar"), ("ajustes", "Ajustes"))
        for i, (chave, texto) in enumerate(itens):
            botao = ctk.CTkButton(
                barra, text=texto, anchor="w", height=44, corner_radius=10,
                font=self._fonte(15, "bold"), fg_color="transparent",
                hover_color=COR_CARTAO_2, text_color=COR_SUAVE,
                command=lambda c=chave: self._mostrar_pagina(c))
            botao.grid(row=2 + i, column=0, sticky="ew", padx=14, pady=3)
            self.botoes_nav[chave] = botao

        rodape = ctk.CTkFrame(barra, fg_color="transparent")
        rodape.grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 20))
        rodape.grid_columnconfigure(0, weight=1)

        self.label_status_camera = self._rotulo(rodape, "●  Câmera desligada", 13, cor=COR_SUAVE)
        self.label_status_camera.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 8))

        self.botao_camera = self._botao(rodape, "▶   Ligar câmera", self._alternar_camera,
                                        "sucesso", height=50, font=self._fonte(15, "bold"))
        self.botao_camera.grid(row=1, column=0, sticky="ew")

    def _mostrar_pagina(self, nome):
        if self._pagina_atual == nome:
            return
        self._pagina_atual = nome
        self.paginas[nome].tkraise()
        for chave, botao in self.botoes_nav.items():
            if chave == nome:
                botao.configure(fg_color=COR_CARTAO_2, text_color=COR_TEXTO)
            else:
                botao.configure(fg_color="transparent", text_color=COR_SUAVE)

        anterior = self._label_video_ativo
        self._label_video_ativo = {"traduzir": self.label_video,
                                   "coletar": self.label_video_coleta}.get(nome)
        if anterior is not None and anterior is not self._label_video_ativo:
            # a PhotoImage e reaproveitada: tira do label antigo para nao
            # mostrar duas vezes o mesmo objeto
            anterior.configure(image="", text=self._texto_video_parado())
            self._foto_video = None

        if nome == "coletar":
            self._atualizar_contagem_coleta()
        elif nome == "treinar":
            self._atualizar_contagens_treino()

    # ------------------------------------------------------------------
    # Pagina: Traduzir
    # ------------------------------------------------------------------

    def _montar_pagina_traduzir(self, pagina):
        pagina.grid_columnconfigure(0, weight=1)
        pagina.grid_columnconfigure(1, weight=0, minsize=320)
        pagina.grid_rowconfigure(1, weight=1)

        cabecalho = ctk.CTkFrame(pagina, fg_color="transparent")
        cabecalho.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        cabecalho.grid_columnconfigure(0, weight=1)
        self._rotulo(cabecalho, "Traduzir", 26, "bold").grid(row=0, column=0, sticky="w")
        self.seletor_modo = ctk.CTkSegmentedButton(
            cabecalho, values=["Letras", "Números"], variable=self.var_modo,
            command=self._ao_trocar_modo, height=36, font=self._fonte(14, "bold"),
            selected_color=COR_DESTAQUE, selected_hover_color=COR_DESTAQUE_HOVER,
            unselected_color=COR_CARTAO, unselected_hover_color=COR_CARTAO_2,
            fg_color=COR_CARTAO)
        self.seletor_modo.grid(row=0, column=1, sticky="e")

        # --- video ---
        cartao_video, self.label_video = self._area_video(pagina)
        cartao_video.grid(row=1, column=0, sticky="nsew", padx=(0, 16))

        # --- painel lateral ---
        painel = ctk.CTkFrame(pagina, fg_color="transparent", width=320)
        painel.grid(row=1, column=1, sticky="nsew")
        painel.grid_columnconfigure(0, weight=1)

        cartao_letra = self._cartao(painel)
        cartao_letra.grid(row=0, column=0, sticky="ew")
        cartao_letra.grid_columnconfigure(0, weight=1)

        self.label_tipo_atual = self._rotulo(cartao_letra, "LETRA", 12, "bold", cor=COR_SUAVE)
        self.label_tipo_atual.grid(row=0, column=0, pady=(16, 0))
        self.label_letra_atual = self._rotulo(cartao_letra, "–", 110, "bold")
        self.label_letra_atual.grid(row=1, column=0, pady=(0, 0))

        linha_conf = ctk.CTkFrame(cartao_letra, fg_color="transparent")
        linha_conf.grid(row=2, column=0, sticky="ew", padx=20)
        linha_conf.grid_columnconfigure(0, weight=1)
        self.barra_confianca = ctk.CTkProgressBar(
            linha_conf, height=8, corner_radius=4, progress_color=COR_DESTAQUE, fg_color=COR_CARTAO_2)
        self.barra_confianca.set(0)
        self.barra_confianca.grid(row=0, column=0, sticky="ew")
        self.label_confianca = self._rotulo(linha_conf, "0%", 13, cor=COR_SUAVE, width=44)
        self.label_confianca.grid(row=0, column=1, padx=(10, 0))

        self.pilula_mao = self._pilula(cartao_letra)
        self.pilula_mao.grid(row=3, column=0, pady=(14, 18))
        self._definir_pilula(self.pilula_mao, "Câmera desligada", COR_SUAVE)

        # --- correcao rapida ---
        cartao_corrigir = self._cartao(painel)
        cartao_corrigir.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        cartao_corrigir.grid_columnconfigure(0, weight=1)
        self._rotulo(cartao_corrigir, "Errou? Toque na letra certa", 13, cor=COR_SUAVE).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 6))
        self.linha_alternativas = ctk.CTkFrame(cartao_corrigir, fg_color="transparent")
        self.linha_alternativas.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        self.botoes_alternativas = []
        for i in range(3):
            self.linha_alternativas.grid_columnconfigure(i, weight=1)
            botao = self._botao(self.linha_alternativas, "–", lambda i=i: self._usar_alternativa(i),
                                "secundario", height=44, font=self._fonte(20, "bold"), state="disabled")
            botao.grid(row=0, column=i, sticky="ew", padx=4)
            self.botoes_alternativas.append(botao)

        # --- sinal com movimento ---
        cartao_mov = self._cartao(painel)
        cartao_mov.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        cartao_mov.grid_columnconfigure(0, weight=1)
        self.botao_gravar_dinamico = self._botao(
            cartao_mov, "Sinal com movimento   J · X · Z · Ç",
            self._iniciar_reconhecimento_dinamico, "secundario", height=46)
        self.botao_gravar_dinamico.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        self.barra_dinamico = ctk.CTkProgressBar(
            cartao_mov, height=6, corner_radius=3, progress_color=COR_ALERTA, fg_color=COR_CARTAO_2)
        self.barra_dinamico.set(0)

        # --- opcoes ---
        cartao_opcoes = self._cartao(painel)
        cartao_opcoes.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ctk.CTkSwitch(cartao_opcoes, text="Falar cada letra", variable=self.var_audio_ligado,
                      font=self._fonte(14), progress_color=COR_DESTAQUE).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 6))
        ctk.CTkSwitch(cartao_opcoes, text="Escrever automaticamente", variable=self.var_auto_adicionar,
                      font=self._fonte(14), progress_color=COR_DESTAQUE).grid(
            row=1, column=0, sticky="w", padx=16, pady=(6, 14))

        # --- texto ---
        cartao_texto = self._cartao(pagina)
        cartao_texto.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        cartao_texto.grid_columnconfigure(0, weight=1)

        self.caixa_texto = ctk.CTkTextbox(
            cartao_texto, height=70, font=self._fonte(28, "bold"), fg_color=COR_CARTAO,
            text_color=COR_TEXTO, border_width=0, wrap="word")
        self.caixa_texto.grid(row=0, column=0, sticky="ew", padx=(16, 8), pady=12)

        botoes = ctk.CTkFrame(cartao_texto, fg_color="transparent")
        botoes.grid(row=0, column=1, sticky="e", padx=(0, 14))
        self._botao(botoes, "Espaço", self._adicionar_espaco, "secundario", width=92).grid(
            row=0, column=0, padx=4)
        self._botao(botoes, "⌫  Apagar", self._apagar_ultima_letra, "secundario", width=104).grid(
            row=0, column=1, padx=4)
        self._botao(botoes, "Limpar", self._limpar_texto, "fantasma", width=84).grid(
            row=0, column=2, padx=4)
        self._botao(botoes, "Falar frase", self._falar_frase_completa, "primario", width=118).grid(
            row=0, column=3, padx=4)

    # ------------------------------------------------------------------
    # Pagina: Coletar
    # ------------------------------------------------------------------

    def _montar_pagina_coletar(self, pagina):
        pagina.grid_columnconfigure(0, weight=1)
        pagina.grid_columnconfigure(1, weight=0, minsize=380)
        pagina.grid_rowconfigure(1, weight=1)

        self._rotulo(pagina, "Coletar amostras", 26, "bold").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        cartao_video, self.label_video_coleta = self._area_video(pagina)
        cartao_video.grid(row=1, column=0, sticky="nsew", padx=(0, 16))

        painel = self._cartao(pagina, width=380)
        painel.grid(row=1, column=1, sticky="nsew")
        painel.grid_columnconfigure(0, weight=1)

        self._rotulo(painel, "Pessoa", 13, cor=COR_SUAVE).grid(
            row=0, column=0, sticky="w", padx=18, pady=(18, 4))
        self.entry_pessoa_coleta = ctk.CTkEntry(
            painel, height=38, corner_radius=10, font=self._fonte(14),
            fg_color=COR_CARTAO_2, border_color=COR_BORDA, placeholder_text="Seu nome")
        self.entry_pessoa_coleta.grid(row=1, column=0, sticky="ew", padx=18)
        self.entry_pessoa_coleta.bind("<KeyRelease>", lambda _e: self._atualizar_contagem_coleta())
        self.entry_pessoa_coleta.bind("<Return>", lambda _e: self.focus_set())

        self.seletor_tipo_coleta = ctk.CTkSegmentedButton(
            painel, values=list(TIPOS_COLETA), variable=self.var_tipo_coleta,
            command=self._ao_trocar_tipo_coleta, height=36, font=self._fonte(14, "bold"),
            selected_color=COR_DESTAQUE, selected_hover_color=COR_DESTAQUE_HOVER,
            unselected_color=COR_CARTAO_2, unselected_hover_color=COR_BORDA, fg_color=COR_CARTAO_2)
        self.seletor_tipo_coleta.grid(row=2, column=0, sticky="ew", padx=18, pady=(16, 12))

        self.grade_letras = ctk.CTkFrame(painel, fg_color="transparent")
        self.grade_letras.grid(row=3, column=0, sticky="ew", padx=14)
        self._montar_grade_letras()

        contador = ctk.CTkFrame(painel, fg_color=COR_CARTAO_2, corner_radius=12)
        contador.grid(row=4, column=0, sticky="ew", padx=18, pady=(16, 0))
        contador.grid_columnconfigure(1, weight=1)
        self.label_letra_coleta = self._rotulo(contador, "A", 44, "bold", width=70)
        self.label_letra_coleta.grid(row=0, column=0, rowspan=2, padx=(12, 6), pady=10)
        self.label_total_coleta = self._rotulo(contador, "0 amostras", 20, "bold", anchor="w")
        self.label_total_coleta.grid(row=0, column=1, sticky="sw", pady=(14, 0))
        self.label_sessao_coleta = self._rotulo(contador, "+0 agora", 13, cor=COR_SUAVE, anchor="w")
        self.label_sessao_coleta.grid(row=1, column=1, sticky="nw", pady=(0, 14))

        self.pilula_coleta = self._pilula(painel)
        self.pilula_coleta.grid(row=5, column=0, pady=(14, 0))
        self._definir_pilula(self.pilula_coleta, "Câmera desligada", COR_SUAVE)

        self.barra_dinamico_coleta = ctk.CTkProgressBar(
            painel, height=6, corner_radius=3, progress_color=COR_ALERTA, fg_color=COR_CARTAO_2)
        self.barra_dinamico_coleta.set(0)

        self.botao_salvar_coleta = self._botao(
            painel, "Salvar amostra", self._salvar_amostra_coleta, "primario",
            height=54, font=self._fonte(16, "bold"))
        self.botao_salvar_coleta.grid(row=7, column=0, sticky="ew", padx=18, pady=(14, 4))
        self.label_atalho_coleta = self._rotulo(painel, "ou aperte a barra de espaço", 12, cor=COR_SUAVE)
        self.label_atalho_coleta.grid(row=8, column=0)

        self.botao_desfazer_coleta = self._botao(
            painel, "Desfazer última", self._desfazer_amostra_coleta, "fantasma", height=36)
        self.botao_desfazer_coleta.grid(row=9, column=0, sticky="ew", padx=18, pady=(10, 0))

        # contribuicao de dados: copia as amostras desta pessoa para
        # contribuicoes/<nome>/, a pasta que vai no Pull Request (ver README)
        self._botao(painel, "Exportar minhas amostras", self._exportar_contribuicao, "fantasma",
                    height=36).grid(row=10, column=0, sticky="ew", padx=18, pady=(8, 18))

    def _montar_grade_letras(self):
        for filho in self.grade_letras.winfo_children():
            filho.destroy()
        self._botoes_letras_coleta = {}
        valores = TIPOS_COLETA[self.var_tipo_coleta.get()]
        colunas = 6 if len(valores) > 5 else len(valores)
        for c in range(6):
            self.grade_letras.grid_columnconfigure(c, weight=1 if c < colunas else 0)
        for i, letra in enumerate(valores):
            botao = ctk.CTkButton(
                self.grade_letras, text=letra, width=46, height=40, corner_radius=10,
                font=self._fonte(16, "bold"), fg_color=COR_CARTAO_2, hover_color=COR_BORDA,
                text_color=COR_TEXTO, command=lambda l=letra: self._selecionar_letra_coleta(l))
            botao.grid(row=i // colunas, column=i % colunas, padx=3, pady=3, sticky="ew")
            self._botoes_letras_coleta[letra] = botao
        self._pintar_letra_selecionada()

    def _pintar_letra_selecionada(self):
        atual = self.var_letra_coleta.get()
        for letra, botao in self._botoes_letras_coleta.items():
            if letra == atual:
                botao.configure(fg_color=COR_DESTAQUE, hover_color=COR_DESTAQUE_HOVER, text_color="white")
            else:
                botao.configure(fg_color=COR_CARTAO_2, hover_color=COR_BORDA, text_color=COR_TEXTO)

    # ------------------------------------------------------------------
    # Pagina: Treinar
    # ------------------------------------------------------------------

    def _montar_pagina_treinar(self, pagina):
        pagina.grid_columnconfigure(0, weight=1)

        self._rotulo(pagina, "Treinar", 26, "bold").grid(row=0, column=0, sticky="w")
        self._rotulo(pagina, "Depois de coletar amostras, treine para o tradutor aprender os sinais novos.",
                     14, cor=COR_SUAVE).grid(row=1, column=0, sticky="w", pady=(2, 16))

        linha_cartoes = ctk.CTkFrame(pagina, fg_color="transparent")
        linha_cartoes.grid(row=2, column=0, sticky="ew")
        self.cartoes_treino = {}
        definicoes = (("letras", "Letras", "Alfabeto sem movimento"),
                      ("numeros", "Números", "0 a 9"),
                      ("dinamico", "Movimento", "J · X · Z · Ç"))
        for i, (chave, titulo, descricao) in enumerate(definicoes):
            linha_cartoes.grid_columnconfigure(i, weight=1, uniform="cartoes")
            cartao = self._cartao(linha_cartoes)
            cartao.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0 if i == 2 else 8))
            cartao.grid_columnconfigure(0, weight=1)
            self._rotulo(cartao, titulo, 20, "bold").grid(row=0, column=0, sticky="w", padx=18, pady=(18, 0))
            self._rotulo(cartao, descricao, 13, cor=COR_SUAVE).grid(row=1, column=0, sticky="w", padx=18)
            label_qtd = self._rotulo(cartao, "– amostras", 15, "bold", cor=COR_TEXTO)
            label_qtd.grid(row=2, column=0, sticky="w", padx=18, pady=(14, 14))

            botao_treinar = self._botao(cartao, "Treinar", lambda c=chave: self._acao_treinar(c),
                                        "primario", height=44)
            botao_treinar.grid(row=3, column=0, sticky="ew", padx=18)
            botao_avaliar = None
            if chave != "dinamico":
                botao_avaliar = self._botao(cartao, "Gerar relatório",
                                            lambda c=chave: self._acao_avaliar(c), "fantasma", height=36)
                botao_avaliar.grid(row=4, column=0, sticky="ew", padx=18, pady=(8, 0))
            label_resultado = self._rotulo(cartao, " ", 13, cor=COR_SUAVE, anchor="w")
            label_resultado.grid(row=5, column=0, sticky="ew", padx=18, pady=(10, 16))
            self.cartoes_treino[chave] = dict(qtd=label_qtd, treinar=botao_treinar,
                                              avaliar=botao_avaliar, resultado=label_resultado)

        status = ctk.CTkFrame(pagina, fg_color="transparent")
        status.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        status.grid_columnconfigure(0, weight=1)
        self.label_status_ferramentas = self._rotulo(status, " ", 14, cor=COR_SUAVE, anchor="w")
        self.label_status_ferramentas.grid(row=0, column=0, sticky="w")
        self.botao_detalhes = self._botao(status, "Ver detalhes", self._alternar_detalhes,
                                          "fantasma", height=32, width=120, font=self._fonte(13))
        self.botao_detalhes.grid(row=0, column=1, sticky="e")
        self.barra_ferramentas = ctk.CTkProgressBar(
            status, mode="indeterminate", height=6, corner_radius=3,
            progress_color=COR_DESTAQUE, fg_color=COR_CARTAO_2)

        self.caixa_log_ferramentas = ctk.CTkTextbox(
            pagina, height=180, font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=COR_CARTAO, corner_radius=12, text_color=COR_SUAVE)
        self.caixa_log_ferramentas.insert("1.0", "Nenhuma tarefa executada ainda.")
        self.caixa_log_ferramentas.configure(state="disabled")
        self._detalhes_visiveis = False

        # --- apagar amostras ---
        cartao_rem = self._cartao(pagina)
        cartao_rem.grid(row=5, column=0, sticky="ew", pady=(16, 0))
        self._rotulo(cartao_rem, "Apagar amostras", 16, "bold").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=18, pady=(14, 8))
        self.var_tipo_remover = ctk.StringVar(value="Letras")
        ctk.CTkSegmentedButton(
            cartao_rem, values=list(TIPOS_REMOVER), variable=self.var_tipo_remover,
            command=self._ao_trocar_tipo_remover, height=34, font=self._fonte(13, "bold"),
            selected_color=COR_DESTAQUE, selected_hover_color=COR_DESTAQUE_HOVER,
            unselected_color=COR_CARTAO_2, unselected_hover_color=COR_BORDA, fg_color=COR_CARTAO_2,
        ).grid(row=1, column=0, sticky="w", padx=(18, 8), pady=(0, 16))
        self.var_letra_remover = ctk.StringVar(value=TIPOS_REMOVER["Letras"][0])
        self.menu_letra_remover = ctk.CTkOptionMenu(
            cartao_rem, values=TIPOS_REMOVER["Letras"], variable=self.var_letra_remover, width=80,
            height=34, corner_radius=10, fg_color=COR_CARTAO_2, button_color=COR_BORDA,
            button_hover_color=COR_DESTAQUE, font=self._fonte(14, "bold"))
        self.menu_letra_remover.grid(row=1, column=1, padx=8, pady=(0, 16))
        self.entry_pessoa_remover = ctk.CTkEntry(
            cartao_rem, placeholder_text="Pessoa (vazio = todas)", width=210, height=34,
            corner_radius=10, fg_color=COR_CARTAO_2, border_color=COR_BORDA)
        self.entry_pessoa_remover.grid(row=1, column=2, padx=8, pady=(0, 16))
        self._botao(cartao_rem, "Apagar", self._acao_remover_clicada, "perigo",
                    width=100, height=34).grid(row=1, column=3, padx=(8, 18), pady=(0, 16))

    # ------------------------------------------------------------------
    # Pagina: Ajustes
    # ------------------------------------------------------------------

    def _montar_pagina_ajustes(self, pagina):
        pagina.grid_columnconfigure(0, weight=1)
        self._rotulo(pagina, "Ajustes", 26, "bold").grid(row=0, column=0, sticky="w", pady=(0, 16))

        # --- camera ---
        cartao_cam = self._cartao(pagina)
        cartao_cam.grid(row=1, column=0, sticky="ew")
        cartao_cam.grid_columnconfigure(1, weight=1)
        self._rotulo(cartao_cam, "Câmera", 16, "bold").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 10))
        estilo_menu = dict(height=36, corner_radius=10, fg_color=COR_CARTAO_2, button_color=COR_BORDA,
                           button_hover_color=COR_DESTAQUE, font=self._fonte(14), width=200)
        self._rotulo(cartao_cam, "Dispositivo", 14, cor=COR_SUAVE).grid(row=1, column=0, sticky="w", padx=18)
        ctk.CTkOptionMenu(cartao_cam, values=[f"Câmera {i}" for i in range(4)],
                          variable=self.var_camera, **estilo_menu).grid(row=1, column=1, sticky="w", padx=8, pady=4)
        self._rotulo(cartao_cam, "Resolução", 14, cor=COR_SUAVE).grid(row=2, column=0, sticky="w", padx=18)
        ctk.CTkOptionMenu(cartao_cam, values=list(RESOLUCOES), variable=self.var_resolucao,
                          **estilo_menu).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ctk.CTkSwitch(cartao_cam, text="Mostrar pontos da mão", variable=self.var_desenhar,
                      command=self._ao_mudar_desenho, font=self._fonte(14),
                      progress_color=COR_DESTAQUE).grid(row=3, column=0, columnspan=2, sticky="w",
                                                        padx=18, pady=(10, 4))
        self._rotulo(cartao_cam, "Trocar câmera ou resolução vale ao ligar a câmera de novo.",
                     12, cor=COR_SUAVE).grid(row=4, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 16))

        # --- reconhecimento ---
        cartao_rec = self._cartao(pagina)
        cartao_rec.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        cartao_rec.grid_columnconfigure(0, weight=1)
        self._rotulo(cartao_rec, "Reconhecimento", 16, "bold").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 6))

        self._rotulo(cartao_rec, "Confiança mínima", 14).grid(row=1, column=0, sticky="w", padx=18)
        self.label_valor_confianca = self._rotulo(cartao_rec, f"{CONFIANCA_MINIMA_PADRAO*100:.0f}%", 14, "bold")
        self.label_valor_confianca.grid(row=1, column=1, sticky="e", padx=18)
        self.slider_confianca = ctk.CTkSlider(
            cartao_rec, from_=0.30, to=0.95, number_of_steps=65, command=self._on_mudar_confianca,
            button_color=COR_DESTAQUE, button_hover_color=COR_DESTAQUE_HOVER, progress_color=COR_DESTAQUE)
        self.slider_confianca.set(CONFIANCA_MINIMA_PADRAO)
        self.slider_confianca.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(6, 0))
        self._rotulo(cartao_rec, "Mais alto: menos letras erradas, mas pede o sinal mais nítido.",
                     12, cor=COR_SUAVE).grid(row=3, column=0, columnspan=2, sticky="w", padx=18, pady=(2, 14))

        self._rotulo(cartao_rec, "Estabilidade", 14).grid(row=4, column=0, sticky="w", padx=18)
        self.label_valor_buffer = self._rotulo(cartao_rec, f"{TAMANHO_BUFFER_PADRAO} quadros", 14, "bold")
        self.label_valor_buffer.grid(row=4, column=1, sticky="e", padx=18)
        self.slider_buffer = ctk.CTkSlider(
            cartao_rec, from_=3, to=15, number_of_steps=12, command=self._on_mudar_buffer,
            button_color=COR_DESTAQUE, button_hover_color=COR_DESTAQUE_HOVER, progress_color=COR_DESTAQUE)
        self.slider_buffer.set(TAMANHO_BUFFER_PADRAO)
        self.slider_buffer.grid(row=5, column=0, columnspan=2, sticky="ew", padx=18, pady=(6, 0))
        self._rotulo(cartao_rec, "Mais alto: escreve com mais calma e oscila menos.",
                     12, cor=COR_SUAVE).grid(row=6, column=0, columnspan=2, sticky="w", padx=18, pady=(2, 16))

        # --- sobre ---
        cartao_sobre = self._cartao(pagina)
        cartao_sobre.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self._rotulo(cartao_sobre, "Sobre", 16, "bold").grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))
        self._rotulo(
            cartao_sobre,
            "Tradutor de Libras em tempo real  ·  TCC em Tecnologia em Sistemas para Internet\n"
            "IFSul – Câmpus Charqueadas  ·  Autor: Rafael",
            13, cor=COR_SUAVE, justify="left").grid(row=1, column=0, sticky="w", padx=18, pady=(0, 16))

    # ==================================================================
    # Avisos rapidos (toast)
    # ==================================================================

    def _avisar(self, texto, cor=COR_TEXTO, duracao_ms=2600):
        self.toast.configure(text=texto, text_color=cor)
        self.toast.place(relx=0.5, rely=1.0, y=-24, anchor="s")
        self.toast.lift()
        if self._after_id_toast is not None:
            self.after_cancel(self._after_id_toast)
        self._after_id_toast = self.after(duracao_ms, self.toast.place_forget)

    # ==================================================================
    # Camera
    # ==================================================================

    def _camera_ligada(self):
        return self.camera is not None and self.camera.ativo

    def _texto_video_parado(self):
        if self._camera_ligada():
            return "Abrindo câmera…"
        return "Câmera desligada\n\nclique em  ▶ Ligar câmera  na barra lateral"

    def _alternar_camera(self):
        if self.camera is None:
            self._ligar_camera()
        else:
            self._desligar_camera()

    def _ligar_camera(self):
        try:
            indice = int(self.var_camera.get().split()[-1])
        except ValueError:
            indice = 0
        resolucao = RESOLUCOES.get(self.var_resolucao.get(), (640, 480))

        with self.lock_motor:
            self.motor.resetar_buffer()
        self.camera = CameraWorker(indice, resolucao, self.motor, self.lock_motor)
        self.camera.iniciar()
        self._tempos_quadros.clear()

        self.botao_camera.configure(text="■   Desligar câmera", fg_color=COR_PERIGO,
                                    hover_color=COR_PERIGO_HOVER)
        self.label_status_camera.configure(text="●  Abrindo câmera…", text_color=COR_ALERTA)
        for label in (self.label_video, self.label_video_coleta):
            label.configure(image="", text="Abrindo câmera…")
        self._foto_video = None

    def _desligar_camera(self, erro=None):
        camera = self.camera
        self.camera = None
        if camera is not None:
            camera.parar()
        if self._gravando_dinamico:
            self._cancelar_gravacao_dinamica()
        with self.lock_motor:
            self.motor.resetar_buffer()
        self._ultimo_info = None
        self._foto_video = None

        self.botao_camera.configure(text="▶   Ligar câmera", fg_color=COR_SUCESSO,
                                    hover_color=COR_SUCESSO_HOVER)
        self.label_status_camera.configure(text="●  Câmera desligada", text_color=COR_SUAVE)
        for label in (self.label_video, self.label_video_coleta):
            label.configure(image="", text=self._texto_video_parado())
        self._mostrar_sem_mao(camera_desligada=True)
        self._definir_pilula(self.pilula_coleta, "Câmera desligada", COR_SUAVE)
        if erro:
            messagebox.showerror("Câmera", erro)

    def _loop_ui(self):
        """Roda na thread da interface: consome os resultados publicados
        pelo CameraWorker. Sempre se reagenda (try/finally) - na versao
        anterior, qualquer erro aqui matava o loop e a camera nao voltava
        mais depois de pausar."""
        try:
            camera = self.camera
            if camera is not None:
                if not camera.ativo:
                    self._desligar_camera(erro=camera.erro or "A câmera foi desconectada.")
                    return
                infos = []
                while camera.saida:
                    infos.append(camera.saida.popleft())
                if infos:
                    self._processar_infos(infos)
        except Exception as e:  # mostra o erro mas mantem o app vivo
            print(f"Aviso: erro na atualizacao da interface: {e}")
        finally:
            self._after_id = self.after(INTERVALO_UI_MS, self._loop_ui)

    def _processar_infos(self, infos):
        agora = time.perf_counter()
        for _ in infos:
            self._tempos_quadros.append(agora)
        if len(self._tempos_quadros) >= 2:
            intervalo = self._tempos_quadros[-1] - self._tempos_quadros[0]
            fps = (len(self._tempos_quadros) - 1) / intervalo if intervalo > 0 else 0
            self.label_status_camera.configure(text=f"●  Câmera ligada · {fps:.0f} fps",
                                               text_color=COR_SUCESSO)

        for info in infos:
            if self._gravando_dinamico:
                self._processar_gravacao_dinamica(info)
            elif info["letra_nova_estavel"] and self._pagina_atual == "traduzir":
                self._ao_detectar_letra_estavel(info["letra_nova_estavel"], info["top3"])

        ultimo = infos[-1]
        self._ultimo_info = ultimo
        self._desenhar_video(ultimo["frame"])
        if not self._gravando_dinamico:
            self._atualizar_status(ultimo)
            self._atualizar_status_coleta(ultimo)

    def _desenhar_video(self, quadro_bgr):
        label = self._label_video_ativo
        if label is None:
            return
        cartao = label._cartao
        largura = cartao.winfo_width() - 20
        altura = cartao.winfo_height() - 20
        if largura < 50 or altura < 50:
            return
        h, w = quadro_bgr.shape[:2]
        escala = min(largura / w, altura / h)
        nova = (max(1, int(w * escala)), max(1, int(h * escala)))
        interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
        quadro = cv2.resize(quadro_bgr, nova, interpolation=interpolacao)
        imagem = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))

        if self._foto_video is not None and self._foto_video.width() == nova[0] \
                and self._foto_video.height() == nova[1]:
            self._foto_video.paste(imagem)
        else:
            self._foto_video = ImageTk.PhotoImage(imagem)
            label.configure(image=self._foto_video, text="")

    # ==================================================================
    # Traduzir
    # ==================================================================

    def _rotulo_modo(self):
        return MODOS[self.modo_reconhecimento]["rotulo_singular"]

    def _mostrar_sem_mao(self, camera_desligada=False):
        self.label_letra_atual.configure(text="–", text_color=COR_SUAVE)
        self.barra_confianca.set(0)
        self.label_confianca.configure(text="0%")
        if camera_desligada:
            self._definir_pilula(self.pilula_mao, "Câmera desligada", COR_SUAVE)
        else:
            self._definir_pilula(self.pilula_mao, "Mostre a mão para a câmera", COR_SUAVE)

    def _atualizar_status(self, info):
        if not info["mao_detectada"]:
            self._mostrar_sem_mao()
            return
        confianca = info["confianca_atual"]
        confiante = confianca >= self.motor.confianca_minima
        cor = COR_SUCESSO if confiante else COR_ALERTA
        self.label_letra_atual.configure(text=str(info["letra_atual"]), text_color=cor)
        self.barra_confianca.configure(progress_color=cor)
        self.barra_confianca.set(confianca)
        self.label_confianca.configure(text=f"{confianca*100:.0f}%")
        if confiante:
            self._definir_pilula(self.pilula_mao, "Reconhecendo", COR_SUCESSO)
        else:
            self._definir_pilula(self.pilula_mao, "Segure o sinal firme", COR_ALERTA)

    def _ao_trocar_modo(self, valor):
        """Alterna o modelo estatico entre Letras e Numeros (o MediaPipe e
        as configuracoes sao mantidos)."""
        modo = "numeros" if valor == "Números" else "letras"
        if modo == self.modo_reconhecimento:
            return
        cfg = MODOS[modo]
        if not Path(cfg["modelo"]).exists():
            messagebox.showwarning(
                "Modelo não treinado",
                f"Ainda não existe modelo de {cfg['nome']}.\n\n"
                "Colete amostras em Coletar e depois clique em Treinar.")
            self.var_modo.set("Números" if self.modo_reconhecimento == "numeros" else "Letras")
            return

        if self._gravando_dinamico:
            self._cancelar_gravacao_dinamica()
        with self.lock_motor:
            self.motor.carregar_modelo(str(cfg["modelo"]))
        self.modo_reconhecimento = modo
        self.label_tipo_atual.configure(text=cfg["rotulo_singular"].upper())
        self._mostrar_sem_mao(camera_desligada=not self._camera_ligada())
        self._definir_alternativas([], None)
        estado = "normal" if modo == "letras" else "disabled"
        self.botao_gravar_dinamico.configure(state=estado)

    def _ao_detectar_letra_estavel(self, letra, top3):
        if self.var_auto_adicionar.get():
            self.caixa_texto.insert("end", letra)
            self._ultima_letra_adicionada = letra
            self._definir_alternativas(top3, letra)
        if self.var_audio_ligado.get():
            self.falante.falar(letra)

    def _definir_alternativas(self, top3, escolhida):
        self._alternativas = [str(c) for c, _p in top3 if str(c) != str(escolhida)][:3]
        for i, botao in enumerate(self.botoes_alternativas):
            if i < len(self._alternativas):
                botao.configure(text=self._alternativas[i], state="normal")
            else:
                botao.configure(text="–", state="disabled")

    def _usar_alternativa(self, indice):
        if indice >= len(self._alternativas):
            return
        nova = self._alternativas[indice]
        conteudo = self.caixa_texto.get("1.0", "end-1c")
        if self._ultima_letra_adicionada and conteudo.endswith(self._ultima_letra_adicionada):
            conteudo = conteudo[: -len(self._ultima_letra_adicionada)]
        self.caixa_texto.delete("1.0", "end")
        self.caixa_texto.insert("1.0", conteudo + nova)
        if self.var_audio_ligado.get():
            self.falante.falar(nova)
        self._ultima_letra_adicionada = nova
        self._definir_alternativas([], None)

    def _adicionar_espaco(self):
        self.caixa_texto.insert("end", " ")
        self._ultima_letra_adicionada = None
        self._definir_alternativas([], None)

    def _apagar_ultima_letra(self):
        conteudo = self.caixa_texto.get("1.0", "end-1c")
        if conteudo:
            self.caixa_texto.delete("1.0", "end")
            self.caixa_texto.insert("1.0", conteudo[:-1])
        self._ultima_letra_adicionada = None
        self._definir_alternativas([], None)

    def _limpar_texto(self):
        self.caixa_texto.delete("1.0", "end")
        self._ultima_letra_adicionada = None
        self._definir_alternativas([], None)

    def _falar_frase_completa(self):
        frase = self.caixa_texto.get("1.0", "end-1c").strip()
        if frase:
            self.falante.falar(frase)

    def _atalho_espaco(self, evento):
        """Barra de espaco: em Coletar salva a amostra; em Traduzir insere
        um espaco. Ignorada quando o foco esta num campo de texto."""
        if isinstance(evento.widget, (tk.Entry, tk.Text)):
            return None
        if self._pagina_atual == "coletar":
            self._salvar_amostra_coleta()
            return "break"
        if self._pagina_atual == "traduzir":
            self._adicionar_espaco()
            return "break"
        return None

    # ==================================================================
    # Ajustes
    # ==================================================================

    def _on_mudar_confianca(self, valor):
        with self.lock_motor:
            self.motor.definir_confianca_minima(valor)
        self.label_valor_confianca.configure(text=f"{valor*100:.0f}%")

    def _on_mudar_buffer(self, valor):
        with self.lock_motor:
            self.motor.definir_tamanho_buffer(valor)
        self.label_valor_buffer.configure(text=f"{int(round(valor))} quadros")

    def _ao_mudar_desenho(self):
        self.motor.desenhar_landmarks = bool(self.var_desenhar.get())

    # ==================================================================
    # Coletar
    # ==================================================================

    def _caminho_coleta(self, letra):
        return CAMINHO_DATASET_DINAMICO if letra in LETRAS_DINAMICAS else caminho_dataset_estatico(letra)

    def _contagens_arquivo(self, caminho):
        """Contagem de amostras por (pessoa, rotulo), lida UMA vez por
        arquivo e depois atualizada em memoria a cada salvar/desfazer (ler
        o CSV inteiro a cada amostra deixava a coleta lenta)."""
        chave = str(caminho)
        if chave not in self._cache_contagens:
            contagens = {}
            try:
                df = pd.read_csv(caminho, usecols=["pessoa", "label"], dtype=str)
                pessoas = df["pessoa"].fillna("").str.strip().str.lower()
                contagens = df.assign(p=pessoas).groupby(["p", "label"]).size().to_dict()
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError):
                pass
            self._cache_contagens[chave] = contagens
        return self._cache_contagens[chave]

    def _ajustar_contagem(self, caminho, pessoa, letra, delta):
        contagens = self._contagens_arquivo(caminho)
        k = (pessoa.strip().lower(), letra)
        contagens[k] = max(0, contagens.get(k, 0) + delta)

    def _atualizar_contagem_coleta(self):
        letra = self.var_letra_coleta.get()
        pessoa = self.entry_pessoa_coleta.get().strip().lower()
        total = self._contagens_arquivo(self._caminho_coleta(letra)).get((pessoa, letra), 0) if pessoa else 0
        self.label_letra_coleta.configure(text=letra)
        self.label_total_coleta.configure(text=f"{total} amostra" + ("" if total == 1 else "s"))
        self.label_sessao_coleta.configure(text=f"+{self._contagem_sessao_coleta} agora")

    def _atualizar_status_coleta(self, info):
        if info["mao_detectada"]:
            self._definir_pilula(self.pilula_coleta, "Mão detectada", COR_SUCESSO)
        else:
            self._definir_pilula(self.pilula_coleta, "Mostre a mão para a câmera", COR_SUAVE)

    def _ao_trocar_tipo_coleta(self, tipo):
        valores = TIPOS_COLETA[tipo]
        self.var_letra_coleta.set(valores[0])
        self._montar_grade_letras()
        self._selecionar_letra_coleta(valores[0])

    def _selecionar_letra_coleta(self, letra):
        if self._gravando_dinamico:
            return
        # tira o foco do campo Pessoa: sem isso, a barra de espaco (atalho
        # para salvar) escreveria espacos no nome em vez de salvar amostras
        self.focus_set()
        self.var_letra_coleta.set(letra)
        self._pintar_letra_selecionada()
        self._pode_desfazer_coleta = False
        self._contagem_sessao_coleta = 0
        dinamica = letra in LETRAS_DINAMICAS
        self.botao_salvar_coleta.configure(text="Gravar movimento" if dinamica else "Salvar amostra")
        self._atualizar_contagem_coleta()

    def _salvar_amostra_coleta(self):
        if self._gravando_dinamico:
            return
        if not self._camera_ligada():
            self._avisar("Ligue a câmera primeiro", COR_ALERTA)
            return
        pessoa = self.entry_pessoa_coleta.get().strip()
        if not pessoa:
            self._avisar("Preencha o nome da pessoa", COR_ALERTA)
            return

        letra = self.var_letra_coleta.get()
        if letra in LETRAS_DINAMICAS:
            self._iniciar_gravacao_dinamica(modo="coleta", letra=letra, pessoa=pessoa)
            return

        info = self._ultimo_info
        if not info or not info["mao_detectada"] or info.get("landmarks_brutos") is None:
            self._avisar("Nenhuma mão na câmera", COR_ALERTA, 1500)
            return

        caminho = caminho_dataset_estatico(letra)
        salvar_amostra(info["landmarks_brutos"], pessoa, letra, caminho=str(caminho),
                       z=info.get("landmarks_z"))
        self._registrar_amostra_salva(caminho, pessoa, letra, dinamica=False)

    def _registrar_amostra_salva(self, caminho, pessoa, letra, dinamica):
        self._ajustar_contagem(caminho, pessoa, letra, +1)
        self._ultima_coleta = (caminho, pessoa, letra, dinamica)
        self._contagem_sessao_coleta += 1
        self._pode_desfazer_coleta = True
        self._atualizar_contagem_coleta()

    def _exportar_contribuicao(self):
        pessoa = self.entry_pessoa_coleta.get().strip()
        if not pessoa:
            self._avisar("Preencha o nome da pessoa", COR_ALERTA)
            return
        try:
            pasta, contagens = exportar_contribuicao(pessoa)
        except Exception as e:
            messagebox.showerror("Exportar amostras", f"Não foi possível exportar: {e}")
            return
        if not contagens:
            self._avisar(f"Nenhuma amostra de {pessoa} encontrada", COR_SUAVE)
            return
        total = sum(contagens.values())
        self._avisar(f"{total} amostras exportadas para contribuicoes/{pasta.name}/", COR_SUCESSO, 4500)

    def _desfazer_amostra_coleta(self):
        if not self._pode_desfazer_coleta or self._ultima_coleta is None:
            self._avisar("Nada para desfazer", COR_SUAVE, 1500)
            return
        caminho, pessoa, letra, dinamica = self._ultima_coleta
        if dinamica:
            removeu = remover_ultima_amostra_dinamica(caminho=str(caminho))
        else:
            removeu = remover_ultima_amostra(caminho=str(caminho))
        if removeu:
            self._ajustar_contagem(caminho, pessoa, letra, -1)
            self._contagem_sessao_coleta = max(0, self._contagem_sessao_coleta - 1)
            self._avisar("Última amostra desfeita", COR_TEXTO, 1500)
        self._pode_desfazer_coleta = False
        self._atualizar_contagem_coleta()

    # ==================================================================
    # Sinais com movimento (J, X, Z, Ç): coleta e reconhecimento
    # ==================================================================

    def _iniciar_gravacao_dinamica(self, modo, letra=None, pessoa=None):
        """Duas fases: 'preparo' (NUM_QUADROS_PREPARO quadros para a pessoa
        se posicionar) e 'gravando' (NUM_QUADROS_JANELA quadros com mao)."""
        if self._gravando_dinamico:
            return
        self._gravando_dinamico = True
        self._modo_gravacao_dinamica = modo
        self._fase_gravacao_dinamica = "preparo"
        self._contagem_gravacao_dinamica = 0
        self._buffer_dinamico = []
        self._pessoa_gravacao_dinamica = pessoa
        self._letra_gravacao_dinamica = letra

        if modo == "coleta":
            for w in (self.botao_salvar_coleta, self.botao_desfazer_coleta, self.seletor_tipo_coleta):
                w.configure(state="disabled")
            self.label_atalho_coleta.grid_remove()
            self.barra_dinamico_coleta.grid(row=6, column=0, sticky="ew", padx=18, pady=(14, 0))
            self.barra_dinamico_coleta.set(0)
        else:
            self.botao_gravar_dinamico.configure(state="disabled")
            self.barra_dinamico.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
            self.barra_dinamico.set(0)

    def _status_gravacao(self, texto, cor, progresso):
        if self._modo_gravacao_dinamica == "coleta":
            self._definir_pilula(self.pilula_coleta, texto, cor)
            self.barra_dinamico_coleta.configure(progress_color=cor)
            self.barra_dinamico_coleta.set(progresso)
        else:
            self._definir_pilula(self.pilula_mao, texto, cor)
            self.barra_dinamico.configure(progress_color=cor)
            self.barra_dinamico.set(progresso)

    def _processar_gravacao_dinamica(self, info):
        if self._fase_gravacao_dinamica == "preparo":
            self._contagem_gravacao_dinamica += 1
            self._status_gravacao("Prepare-se…", COR_ALERTA,
                                  self._contagem_gravacao_dinamica / NUM_QUADROS_PREPARO)
            if self._contagem_gravacao_dinamica >= NUM_QUADROS_PREPARO:
                self._fase_gravacao_dinamica = "gravando"
                self._contagem_gravacao_dinamica = 0
                self._buffer_dinamico = []
            return

        # so contam quadros com mao, para a janela nao ter "buracos"
        if info["mao_detectada"] and info.get("landmarks_brutos") is not None:
            self._buffer_dinamico.append(info["landmarks_brutos"])
        self._status_gravacao("Faça o sinal agora!", COR_PERIGO,
                              len(self._buffer_dinamico) / NUM_QUADROS_JANELA)
        if len(self._buffer_dinamico) >= NUM_QUADROS_JANELA:
            self._finalizar_gravacao_dinamica()

    def _encerrar_estado_gravacao(self):
        modo = self._modo_gravacao_dinamica
        self._gravando_dinamico = False
        self._fase_gravacao_dinamica = None
        self._buffer_dinamico = []
        if modo == "coleta":
            for w in (self.botao_salvar_coleta, self.botao_desfazer_coleta, self.seletor_tipo_coleta):
                w.configure(state="normal")
            self.barra_dinamico_coleta.grid_remove()
            self.label_atalho_coleta.grid()
        elif modo == "reconhecer":
            if self.modo_reconhecimento == "letras":
                self.botao_gravar_dinamico.configure(state="normal")
            self.barra_dinamico.grid_remove()
        return modo

    def _finalizar_gravacao_dinamica(self):
        sequencia = self._buffer_dinamico[:NUM_QUADROS_JANELA]
        modo = self._encerrar_estado_gravacao()
        if modo == "coleta":
            pessoa, letra = self._pessoa_gravacao_dinamica, self._letra_gravacao_dinamica
            salvar_amostra_dinamica(sequencia, pessoa, letra, caminho=str(CAMINHO_DATASET_DINAMICO))
            self._registrar_amostra_salva(CAMINHO_DATASET_DINAMICO, pessoa, letra, dinamica=True)
            self._definir_pilula(self.pilula_coleta, f"Amostra de {letra} salva", COR_SUCESSO)
        else:
            self._reconhecer_sequencia_dinamica(sequencia)

    def _cancelar_gravacao_dinamica(self):
        self._encerrar_estado_gravacao()

    def _iniciar_reconhecimento_dinamico(self):
        if not self._camera_ligada():
            self._avisar("Ligue a câmera primeiro", COR_ALERTA)
            return
        if self.modelo_dinamico is None:
            messagebox.showwarning(
                "Modelo não treinado",
                "Ainda não existe modelo de sinais com movimento.\n\n"
                "Grave amostras em Coletar › Movimento e clique em Treinar.")
            return
        self._iniciar_gravacao_dinamica(modo="reconhecer")

    def _reconhecer_sequencia_dinamica(self, sequencia):
        entrada = normalizar_sequencia(achatar_sequencia_bruta(sequencia))
        probs = self.modelo_dinamico.predict_proba([entrada])[0]
        ordem = probs.argsort()[::-1]
        top3 = [(self.modelo_dinamico.classes_[i], float(probs[i])) for i in ordem[:3]]
        letra, confianca = top3[0]

        self.label_letra_atual.configure(text=str(letra), text_color=COR_SUCESSO)
        self.barra_confianca.configure(progress_color=COR_SUCESSO)
        self.barra_confianca.set(confianca)
        self.label_confianca.configure(text=f"{confianca*100:.0f}%")
        self._definir_pilula(self.pilula_mao, f"Movimento: {letra}", COR_SUCESSO)
        self._ao_detectar_letra_estavel(letra, top3)

    # ==================================================================
    # Treinar
    # ==================================================================

    def _atualizar_contagens_treino(self):
        caminhos = {"letras": MODOS["letras"]["csv"], "numeros": MODOS["numeros"]["csv"],
                    "dinamico": CAMINHO_DATASET_DINAMICO}
        for chave, caminho in caminhos.items():
            total = sum(self._contagens_arquivo(caminho).values())
            texto = f"{total:,} amostras".replace(",", ".")
            self.cartoes_treino[chave]["qtd"].configure(text=texto)

    def _alternar_detalhes(self):
        self._detalhes_visiveis = not self._detalhes_visiveis
        if self._detalhes_visiveis:
            self.caixa_log_ferramentas.grid(row=4, column=0, sticky="ew", pady=(10, 0))
            self.botao_detalhes.configure(text="Ocultar detalhes")
        else:
            self.caixa_log_ferramentas.grid_remove()
            self.botao_detalhes.configure(text="Ver detalhes")

    def _botoes_ferramentas_estado(self, ativo):
        estado = "normal" if ativo else "disabled"
        for cartao in self.cartoes_treino.values():
            cartao["treinar"].configure(state=estado)
            if cartao["avaliar"] is not None:
                cartao["avaliar"].configure(state=estado)

    def _rodar_tarefa_ferramenta(self, chave, tipo, descricao, funcao):
        self.label_status_ferramentas.configure(text=f"{descricao}… pode levar alguns segundos.",
                                                text_color=COR_TEXTO)
        self.cartoes_treino[chave]["resultado"].configure(text="Em andamento…", text_color=COR_ALERTA)
        self._botoes_ferramentas_estado(ativo=False)
        self.barra_ferramentas.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.barra_ferramentas.start()

        def alvo():
            buffer = io.StringIO()
            erro = None
            try:
                with contextlib.redirect_stdout(buffer):
                    funcao()
            except Exception as e:
                erro = str(e)
            self._fila_ferramentas.put((chave, tipo, descricao, buffer.getvalue(), erro))

        threading.Thread(target=alvo, daemon=True).start()
        self._verificar_fila_ferramentas()

    def _verificar_fila_ferramentas(self):
        try:
            chave, tipo, descricao, saida, erro = self._fila_ferramentas.get_nowait()
        except queue.Empty:
            self._after_id_ferramentas = self.after(300, self._verificar_fila_ferramentas)
            return
        self._after_id_ferramentas = None

        self.barra_ferramentas.stop()
        self.barra_ferramentas.grid_remove()
        self._botoes_ferramentas_estado(ativo=True)

        label_resultado = self.cartoes_treino[chave]["resultado"]
        falhou = bool(erro) or any(l.strip().lower().startswith("erro") for l in (saida or "").splitlines())
        if falhou:
            saida = (saida or "") + (f"\n\n--- ERRO ---\n{erro}" if erro else "")
            label_resultado.configure(text="✕  Não concluído", text_color=COR_PERIGO)
            self.label_status_ferramentas.configure(text=f"{descricao}: algo deu errado. Veja os detalhes.",
                                                    text_color=COR_PERIGO)
        elif tipo == "treinar":
            resumo = resumo_da_saida(saida) or "Concluído"
            label_resultado.configure(text=f"✓  {resumo}", text_color=COR_SUCESSO)
            self.label_status_ferramentas.configure(text=f"{descricao}: pronto! O tradutor já usa o modelo novo.",
                                                    text_color=COR_SUCESSO)
        else:
            label_resultado.configure(text="✓  Relatório gerado em models/", text_color=COR_SUCESSO)
            self.label_status_ferramentas.configure(text=f"{descricao}: pronto!", text_color=COR_SUCESSO)

        self.caixa_log_ferramentas.configure(state="normal")
        self.caixa_log_ferramentas.delete("1.0", "end")
        self.caixa_log_ferramentas.insert("1.0", saida or "(sem saída)")
        self.caixa_log_ferramentas.configure(state="disabled")

        if tipo == "treinar" and not falhou:
            if chave in ("letras", "numeros") and chave == self.modo_reconhecimento:
                with self.lock_motor:
                    self.motor.carregar_modelo(str(MODOS[chave]["modelo"]))
            elif chave == "dinamico":
                self._carregar_modelo_dinamico()

    def _acao_treinar(self, chave):
        if chave == "dinamico":
            self._rodar_tarefa_ferramenta(chave, "treinar", "Treinando Movimento",
                                          lambda: TreinadorDinamicoIA().treinar())
        else:
            nome = MODOS[chave]["nome"].replace("Numeros", "Números")
            self._rodar_tarefa_ferramenta(chave, "treinar", f"Treinando {nome}",
                                          lambda: TreinadorIA().treinar(modo=chave))

    def _acao_avaliar(self, chave):
        nome = MODOS[chave]["nome"].replace("Numeros", "Números")
        self._rodar_tarefa_ferramenta(chave, "avaliar", f"Gerando relatório de {nome}",
                                      lambda: avaliar_modelos_main(modo=chave))

    def _ao_trocar_tipo_remover(self, tipo):
        valores = TIPOS_REMOVER[tipo]
        self.menu_letra_remover.configure(values=valores)
        self.var_letra_remover.set(valores[0])

    def _acao_remover_clicada(self):
        letra = self.var_letra_remover.get()
        pessoa = self.entry_pessoa_remover.get().strip() or None
        dinamico = self.var_tipo_remover.get() == "Movimento"
        caminho = CAMINHO_DATASET_DINAMICO if dinamico else caminho_dataset_estatico(letra)

        quem = f"de {pessoa}" if pessoa else "de TODAS as pessoas"
        tipo = " (com movimento)" if dinamico else ""
        if not messagebox.askyesno("Apagar amostras",
                                   f"Apagar as amostras de {letra}{tipo} {quem}?\n\n"
                                   "Um backup do arquivo é criado antes."):
            return
        try:
            qtd, backup = remover_amostras_dataset(letra, pessoa, csv_path=str(caminho))
        except FileNotFoundError:
            self._avisar("Ainda não há amostras desse tipo", COR_SUAVE)
            return
        except ValueError as e:
            messagebox.showerror("Erro ao apagar", str(e))
            return

        if qtd == 0:
            self._avisar("Nenhuma amostra encontrada", COR_SUAVE)
            return
        self._cache_contagens.pop(str(caminho), None)
        self._atualizar_contagens_treino()
        self._atualizar_contagem_coleta()
        self._avisar(f"{qtd} amostra(s) apagada(s) · backup em data/", COR_SUCESSO, 3500)

    # ==================================================================
    # Encerramento
    # ==================================================================

    def _ao_fechar(self):
        for nome in ("_after_id", "_after_id_ferramentas", "_after_id_toast"):
            after_id = getattr(self, nome, None)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except (ValueError, tk.TclError):
                    pass
        if self.camera is not None:
            self.camera.parar()
            self.camera = None
        if self.motor is not None:
            self.motor.fechar()
        self.falante.parar()
        self.destroy()


if __name__ == "__main__":
    app = AplicativoLibras()
    app.mainloop()
