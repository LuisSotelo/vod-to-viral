# 🚀 Twitch VOD Auto Clipper

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-active-success)]()
![FFmpeg](https://img.shields.io/badge/ffmpeg-required-orange)
![CUDA](https://img.shields.io/badge/GPU-optional-green)

Convierte automáticamente streams de Twitch en **clips virales listos para TikTok, Reels y Shorts** usando IA.

> 🎯 De VOD largo → contenido viral en minutos

## 📸 Ejemplo

<img src="assets/demo.gif" width="300">

🔥 Detecta momentos hype  
🎧 Transcribe con Whisper  
💬 Analiza chat  
📱 Genera formato vertical (9:16)  
🎯 Subtítulos dinámicos tipo MrBeast  

---

## 💡 Use cases

- Creadores de contenido
- Streamers
- Agencias de marketing
- Automatización de clips virales

---

## 🎬 Demo

**Flujo completo:**

1. Descarga el último VOD de un canal de Twitch  
2. Detecta momentos importantes usando:
   - 🔊 audio (energía/emoción)
   - 🧾 texto (keywords)
   - 💬 chat (reacciones)
   - 🎥 movimiento visual  
3. Genera clips automáticamente:
   - Horizontal (16:9)
   - Vertical (9:16)
4. Añade subtítulos sincronizados estilo viral  

---

## 🧠 Cómo funciona

El sistema usa un enfoque de **multi-signal scoring**:

- 🔊 **Audio peaks** → emociones, gritos, hype  
- 💬 **Chat spikes** → mensajes por segundo  
- 🧾 **Whisper** → palabras clave  
- 🎥 **Movimiento visual** → cambios de escena  

Todos los factores se combinan para detectar los mejores momentos del stream.

---

## ⚡ Features

- 🧠 Detección inteligente de clips (multi-signal scoring)
- 🎧 Transcripción con **Whisper (GPU opcional)**
- 💬 Análisis de chat de Twitch
- 📱 Layout vertical optimizado (9:16)
- 🎨 Subtítulos estilo viral (word-by-word)
- ⚡ Render rápido con ffmpeg
- 🧪 Modo preview
- 🧹 Limpieza automática

---

## 📦 Requisitos

- Python **3.11**
- ffmpeg en PATH
- TwitchDownloaderCLI en PATH
- GPU opcional (CUDA)

---

## 🔧 Instalación

```bash
git clone https://github.com/LuisSotelo/vod-to-viral.git
cd vod-to-viral

py -3.11 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 📦 requirements.txt

El proyecto usa las siguientes dependencias de Python:

```txt
# =========================
# Core
# =========================
python-dotenv
requests==2.32.3
charset-normalizer==3.3.2

yt-dlp>=2024.3.10
opencv-python-headless
numpy
pandas
scipy
tqdm

# =========================
# AI / Audio / Whisper
# =========================
torch
faster-whisper
librosa>=0.10.1
soundfile

# =========================
# Subtítulos
# =========================
pysrt
srt

# =========================
# Utilidades
# =========================
python-dateutil
colorlog
beautifulsoup4
lxml

# =========================
# Opcionales PRO (recomendado)
# =========================
orjson
rich
```

---

## ⚙️ Configuración

Crea .env

```bash
TWITCH_CLIENT_ID=
TWITCH_CLIENT_SECRET=
TWITCH_USER_LOGIN=
OUTPUT_DIR=./out
```

## ▶️ Uso

```bash
python main.py
Con flags:
python main.py --max-clips 10 --vertical
```
---

## 📁 Estructura del proyecto

```bash
vod-to-viral/
├── main.py
├── requirements.txt
├── .env.example
├── README.md
└── out/
```
---

## 🎥 Output
El sistema genera:

🎬 Clips horizontales (YouTube)
📱 Clips verticales (TikTok / Shorts / Reels)
💬 Subtítulos sincronizados

Ejemplo:
```bash
output/
├── clip_01_vertical.mp4
├── clip_01_horizontal.mp4
└── subtitles/
```
---

## 🧪 Flags útiles
```bash
--max-clips
--min-clip-sec
--max-clip-sec
--vertical
--skip-download
--vod-id
--model-size
--peak-height
--min-peak-distance-sec
--reset
--reset-only
--preview-vertical
```
---

## ⚠️ Notas

-Whisper en CPU puede ser lento

-GPU mejora muchísimo el rendimiento

-Clips dependen de la calidad del VOD y actividad del chat

---

## 🚀 Roadmap
 - [ ]Auto upload a TikTok / YouTube Shorts

 - [ ]Dashboard web (Next.js 👀)

 - [ ]Detección de caras (face tracking)

 - [ ]IA para detectar highlights tipo “rage / clutch”

 - [ ]Integración con OBS / streams en vivo
---

## 🤝 Contribuciones
 Pull requests bienvenidos 🙌

-Fork

-Crea tu rama

-Haz cambios

-PR

---

## 📜 Licencia

 MIT

---

## 👨‍💻 Autor
 Luis Sotelo / LuisHongo

🎥 Twitch

💻 Developer

🚀 Builder de herramientas virales

---

## ⭐ Si te sirve

 Dale estrella al repo ⭐ y compártelo 🔥

---

### 📝 Notas importantes

- 🧠 **PyTorch (torch)**  
  Se instala automáticamente desde `requirements.txt`.  
  Sin embargo, si deseas usar **GPU con CUDA**, puede ser necesario reinstalarlo con la versión adecuada para tu sistema.  
  👉 Consulta: https://pytorch.org/get-started/locally/

- 🎬 **ffmpeg y TwitchDownloaderCLI**  
  Estas herramientas **NO se instalan con pip**.  
  Debes instalarlas manualmente y asegurarte de que estén disponibles en tu `PATH`.

  Ejemplo:
  ```bash
  ffmpeg -version
  TwitchDownloaderCLI --help