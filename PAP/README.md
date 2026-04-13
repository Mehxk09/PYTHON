# Signify
O Signify é uma aplicação web que reconhece gestos de língua gestual em tempo real através da tua webcam e converte-os em texto.

Suporta:

Modo letras (ASL): sinais de mão estáticos para letras (A-Y)
Modo palavras (LGP): gestos dinâmicos para um pequeno vocabulário inicial
Página de dicionário: exemplos visuais (imagens para letras e vídeos para palavras)

Funcionalidades
Transmissão da webcam em tempo real no navegador
Deteção de pontos-chave da mão com MediaPipe (21 pontos)
Dois modelos de IA:
Modelo denso para classificação de letras em ASL
Modelo LSTM para classificação de sequências de palavras
Previsão em direto com barra de confiança
Controlo de texto: limpar, adicionar espaço e apagar
Ajuda integrada na aplicação (?) com ecrã inicial explicativo
Processamento local (sem envio de vídeo para serviços externos)

---

##Tecnologias utilizadas


- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow/Keras, NumPy, scikit-learn
- **Frontend:** HTML, CSS, JavaScript

---

## Estrutura 


PAP/
+-- app.py
+-- dicionario.py
+-- collect_samples.py
+-- collect_words.py
+-- view_samples.py
+-- test_prediction.py
+-- requirements.txt
+-- dataset/
   +-- asl_alphabets/
   +-- psl_words/
+-- model/
   +-- train_model.py
   +-- train_word_model.py
   +-- letter_classifier.py
   +-- *.h5 / *.pkl
+-- templates/
   +-- index.html
   +-- dicionario.html
+-- static/
    +-- style.css
    +-- dicionario.css
    +-- script.js
```


## Como instalar

### 1) Clonar o repositório


git clone https://github.com/Mehxk09/PYTHON.git
cd PYTHON/PAP


### 2) Criar e ativar um ambiente virtual (recomendado)


Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Instalar dependências


```bash
pip install -r requirements.txt
```

---

## Executar a aplicação


```bash
python app.py
```

Abrir o URL local mostrado no terminal (normalmente `http://127.0.0.1:5000`).

---

## Recolha de Dados

### Recolher amostras de letras (ASL)

```bash
python collect_samples.py A
```

-Premir Espaço para guardar a imagem
-Premir q para sair

### Recolher amostras de palavras (LGP)


```bash
python collect_words.py "Boa Tarde"
```

- Premir s para parar e guardar
-Premir Espaço para guardar a imagem
-Premir q para sair

---

## Treinar Modelos

### Treinar o modelo de letras

```bash
python model/train_model.py
```

### Treinar o modelo de palavras

```bash
python model/train_word_model.py
```


## Dicionário

O dicionário está disponível no botão **Dicionário** no canto superior direito da página principal.

Mostra:

* Exemplos em imagem do alfabeto ASL
* Exemplos em vídeo de palavras LGP

---