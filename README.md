# Sistema de Reconhecimento de LIBRAS em Tempo Real 🤟

Este projeto consiste em um sistema inteligente capaz de capturar, treinar e reconhecer sinais da Língua Brasileira de Sinais (LIBRAS) em tempo real por meio da webcam. Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) para o curso de Tecnologia em Internet Systems do IFSUL Câmpus Charqueadas.

## 🚀 Como o Sistema Funciona

O fluxo do projeto é dividido em três etapas principais:
1. **Coleta de Dados (`coletor.py`):** Utiliza o OpenCV para abrir a câmera e o MediaPipe para extrair os 21 pontos tridimensionais (coordenadas geométricas) da mão do usuário, salvando-os em um arquivo `dados.csv`.
2. **Treinamento da IA (`treinamento.py`):** Consome o arquivo de dados utilizando a biblioteca Pandas e treina um classificador de Aprendizado de Máquina (*Random Forest* via Scikit-Learn), gerando o modelo binário preditivo.
3. **Interface de Tradução (`main.py`):** Inicializa a webcam, extrai as coordenadas em tempo real e consulta o modelo treinado para exibir a letra correspondente diretamente no monitor, contando com retorno em áudio via `pyttsx3`.

---

## 🛠️ Pré-requisitos e Tecnologias

* **Linguagem:** Python 3.11.9
* **Bibliotecas Principais:**
  * OpenCV (`opencv-python`)
  * MediaPipe (versão estável `0.10.14`)
  * Scikit-Learn (`scikit-learn`)
  * Pandas (`pandas`)
  * Pyttsx3 (`pyttsx3`)

---

## 📦 Como Instalar e Configurar

1. Certifique-se de estar utilizando o ambiente com Python 3.11 instalado.
2. Instale as dependências executando os comandos abaixo no terminal:

pip install opencv-python pandas scikit-learn pyttsx3
pip install mediapipe==0.10.14 --no-cache-dir

---

## 🎮 Como Executar o Projeto

Siga a ordem lógica de execução os scripts dentro da pasta raiz do projeto:

### 1. Capturar Amostras (Coleta)
Abra o script `scripts/coletor.py`, defina a letra que deseja coletar na última linha (`app.coletor(letra="A")`) e execute:

python scripts/coletor.py

* **Comando:** Posicione a mão na tela fazendo o sinal e pressione a tecla **`S`** repetidamente para salvar variações de posição (recomenda-se de 100 a 200 amostras por letra).
* **Sair:** Pressione **`Q`** para fechar a câmera.

### 2. Treinar o Modelo de IA
Após coletar os dados de mais de uma letra no seu banco de dados, processe o treinamento do algoritmo rodando:

python scripts/treinamento.py

Isso gerará o arquivo do modelo na pasta `models/`.

### 3. Rodar o Tradutor em Tempo Real
Para abrir a câmera final e ver o sistema traduzindo os sinais e vocalizando o áudio em tempo real, execute:

python scripts/main.py

* **Sair:** Pressione **`Q`** na janela de vídeo para encerrar a aplicação.

---

## 📂 Estrutura de Pastas

sistema-reconhecimento/
│
├── data/
│   └── dados.csv               # Banco de dados geométricos gerado
│
├── models/
│   └── modelo.pkl              # Modelo preditivo da IA treinado
│
├── scripts/
│   ├── coletor.py              # Script de captura de dados
│   ├── treinamento.py          # Script de machine learning
│   └── main.py                 # Script de execução em tempo real
│
└── README.md                   # Documentação do projeto

---

## 📝 Autoria

* **Desenvolvedor:** Rafael
* **Instituição:** Instituto Federal de Educação, Ciência e Tecnologia de Sul-rio-grandense (IFSUL) - Câmpus Charqueadas.