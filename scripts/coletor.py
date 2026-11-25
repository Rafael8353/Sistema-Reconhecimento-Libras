"""
coletor.py

Rotina de coleta de dados: abre a webcam, usa o MediaPipe Hands para
detectar a mao e, a cada tecla 'S' pressionada, salva os 21 pontos
(42 coordenadas x,y brutas) no data/dados.csv, junto com a letra e a
pessoa que esta fazendo o gesto.

Principais correcoes em relacao a versao anterior:
  - Antes o codigo chamava cv2.waitKey(1) duas vezes por quadro (uma
    dentro do "if" que so roda quando uma mao e detectada, outra fora),
    o que podia fazer a tecla 'S' ser perdida em alguns quadros. Agora
    so existe UMA chamada de waitKey por quadro.
  - Antes era preciso editar a ultima linha do arquivo
    (app.coletor(letra="Z")) toda vez que fosse coletar uma letra
    diferente, o que e facil de esquecer e pode gerar amostras
    rotuladas errado (voce estar fazendo o sinal de uma letra mas o
    script salvando outra). Agora a letra e a pessoa sao passadas por
    linha de comando (ou perguntadas no terminal se nao forem
    passadas).
  - Agora o script mostra na propria janela da camera a letra atual, a
    pessoa e quantas amostras ja foram salvas nesta sessao, e tem uma
    tecla 'R' para desfazer a ultima captura (caso voce perceba que
    salvou um gesto errado por engano).
  - O dataset agora guarda a coluna 'pessoa' (quem fez o gesto), para
    permitir futuramente treinar/avaliar com dados de varias pessoas
    (pedido do orientador) e analisar se o modelo generaliza entre
    usuarios diferentes.

Atualizacao (TCC 2): agora tambem salva a coordenada z de cada ponto
(profundidade relativa ao pulso, devolvida pelo MediaPipe), colunas
p0_z ... p20_z. Ela e necessaria para separar sinais que so diferem pela
posicao do polegar em relacao ao indicador (F/T, M/N, G/L). CSVs antigos
(sem z) sao migrados automaticamente na primeira gravacao: as colunas z
ficam vazias nas linhas antigas e um backup e criado. Tambem aceita
numerais (0-9), que vao para data/dados_numeros.csv.

Uso:
    python scripts/coletor.py --letra A --pessoa Rafael
    python scripts/coletor.py --letra 5 --pessoa Rafael   # numeral (vai para dados_numeros.csv)
    python scripts/coletor.py            # pergunta letra e pessoa no terminal

Comandos durante a captura:
    S -> salva o gesto atual como uma amostra da letra escolhida
    R -> remove (desfaz) a ultima amostra salva nesta sessao
    Q -> encerra a coleta e grava tudo em data/dados.csv
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from modos import MODOS, modo_do_rotulo
from normalizacao import COLUNAS_XY, COLUNAS_Z

COLUNAS_COORDS = COLUNAS_XY + COLUNAS_Z          # 42 (x,y) + 21 (z)
COLUNAS_FINAIS = COLUNAS_COORDS + ["pessoa", "label"]


def garantir_schema_z(caminho):
    """Se o CSV existente ainda nao tem as colunas z (formato anterior ao
    TCC 2), reescreve o arquivo com as colunas z vazias e as colunas na
    ordem nova, depois de salvar um backup. Sem isso, anexar linhas novas
    (com z) a um CSV antigo corromperia o arquivo."""
    caminho = Path(caminho)
    if not caminho.exists() or caminho.stat().st_size == 0:
        return False
    df = pd.read_csv(caminho, dtype={"label": str, "pessoa": str})
    if all(c in df.columns for c in COLUNAS_Z) and list(df.columns) == COLUNAS_FINAIS:
        return False
    for c in COLUNAS_Z:
        if c not in df.columns:
            df[c] = float("nan")
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = caminho.with_name(f"{caminho.stem}_backup_pre_z_{carimbo}.csv")
    shutil.copy2(caminho, backup)
    df[COLUNAS_FINAIS].to_csv(caminho, index=False)
    print(f"Dataset migrado para o formato com z. Backup em: {backup}")
    return True


def _linha(coordenadas_xy, z, pessoa, letra):
    """Monta uma linha do CSV; z=None grava as colunas z vazias."""
    z_valores = list(z) if z is not None else [float("nan")] * len(COLUNAS_Z)
    return list(coordenadas_xy) + z_valores + [pessoa, letra]


class ColetorDados:
    def __init__(self):
        # Inicializa o "motor" de busca de maos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.lista_final = []

    def coletar(self, letra, pessoa, caminho=None):
        caminho = caminho or str(MODOS[modo_do_rotulo(letra)]["csv"])
        self.caminho = caminho
        cap = cv2.VideoCapture(0)
        print(f"--- Coletando para a Letra: {letra} (pessoa: {pessoa}) ---")
        print("Comandos: 'S' salva | 'R' desfaz a ultima amostra | 'Q' sai e grava")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # Espelha para facilitar o seu movimento
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = self.hands.process(img_rgb)

            mao_detectada = False
            coordenadas_atuais = None
            z_atual = None

            if resultado.multi_hand_landmarks:
                for hand_lms in resultado.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                    coordenadas = []
                    z_valores = []
                    for lm in hand_lms.landmark:
                        coordenadas.extend([lm.x, lm.y])
                        z_valores.append(lm.z)
                    coordenadas_atuais = coordenadas
                    z_atual = z_valores
                    mao_detectada = True

            # --- HUD (informacoes na tela) ---
            cor_status = (0, 255, 0) if mao_detectada else (0, 0, 255)
            cv2.putText(frame, f"Letra: {letra}  Pessoa: {pessoa}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"Amostras salvas: {len(self.lista_final)}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            status_txt = "Mao detectada (S para salvar)" if mao_detectada else "Nenhuma mao detectada"
            cv2.putText(frame, status_txt, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)

            cv2.imshow("TCC Libras - Coletor", frame)

            # Uma unica leitura de teclado por quadro (evita perder teclas)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord('s'):
                if mao_detectada:
                    self.lista_final.append(_linha(coordenadas_atuais, z_atual, pessoa, letra))
                    print(f"Capturado! Total: {len(self.lista_final)}")
                else:
                    print("Nenhuma mao detectada neste quadro, nada foi salvo.")
            elif tecla == ord('r'):
                if self.lista_final:
                    self.lista_final.pop()
                    print(f"Ultima amostra removida. Total agora: {len(self.lista_final)}")
                else:
                    print("Nao ha amostras para remover.")
            elif tecla == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.salvar_csv(caminho)

    def salvar_csv(self, caminho="data/dados.csv"):
        if not self.lista_final:
            print("Nenhuma amostra nova para salvar.")
            return

        garantir_schema_z(caminho)
        df = pd.DataFrame(self.lista_final, columns=COLUNAS_FINAIS)
        arquivo_existe = os.path.exists(caminho)
        df.to_csv(caminho, mode='a', index=False, header=not arquivo_existe)
        print(f"{len(df)} amostras novas salvas em '{caminho}' com sucesso!")


def salvar_amostra(coordenadas, pessoa, letra, caminho="data/dados.csv", z=None):
    """Salva UMA amostra diretamente no CSV. Usado pela tela 'Coletar'
    da GUI (gui_app.py), que grava cada amostra na hora (em vez de acumular
    em memoria e salvar tudo so no final, como o modo terminal acima faz) -
    assim nada se perde se a janela for fechada no meio da coleta.

    `coordenadas` sao os 42 valores (x, y) brutos e `z` os 21 valores de
    profundidade (opcional; sem z as colunas z ficam vazias)."""
    garantir_schema_z(caminho)
    linha = pd.DataFrame([_linha(coordenadas, z, pessoa, letra)], columns=COLUNAS_FINAIS)
    arquivo_existe = os.path.exists(caminho)
    linha.to_csv(caminho, mode='a', index=False, header=not arquivo_existe)


def remover_ultima_amostra(caminho="data/dados.csv"):
    """Remove a ULTIMA linha do CSV. Usado pelo botao 'Desfazer ultima
    amostra' da tela 'Coletar' da GUI, para o caso de ter salvo uma
    amostra por engano. Retorna True se removeu algo, False se o arquivo
    nao existe ou esta vazio."""
    if not os.path.exists(caminho):
        return False
    df = pd.read_csv(caminho, dtype={"label": str, "pessoa": str})
    if df.empty:
        return False
    df.iloc[:-1].to_csv(caminho, index=False)
    return True


def _parse_args():
    parser = argparse.ArgumentParser(description="Coleta amostras de landmarks da mao para o alfabeto de Libras.")
    parser.add_argument("--letra", help="Letra/numero que sera coletado (ex: A ou 5)")
    parser.add_argument("--pessoa", help="Nome da pessoa que esta fazendo o gesto")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    letra = args.letra or input("Digite a letra que sera coletada (ex: A): ").strip().upper()
    pessoa = args.pessoa or input("Digite o nome da pessoa que esta fazendo o gesto: ").strip()

    if not letra or not pessoa:
        raise SystemExit("Letra e pessoa sao obrigatorias.")

    app = ColetorDados()
    app.coletar(letra=letra, pessoa=pessoa)
