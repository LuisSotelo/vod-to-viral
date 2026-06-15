# 🚀 Twitch VOD Auto Clipper & Publisher (Lean AI Edition)

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-active-success)]()
![FFmpeg](https://img.shields.io/badge/ffmpeg-required-orange)
![CUDA](https://img.shields.io/badge/GPU-optional-green)

Convierte automáticamente streams completos de **Twitch y videos de YouTube** en **clips virales y publícalos en YouTube Shorts, TikTok e Instagram Reels** de forma 100% automatizada e inteligente usando IA Multimodal.

> 🎯 **Modelo Lean & Hands-Free:** Olvídate de descargar VODs masivos de 4 horas. El sistema analiza de forma ultra ligera el audio y el chat, descarga quirúrgicamente solo los fragmentos de mayor impacto, genera subtítulos dinámicos y Gemini redacta los títulos y copys virales por ti.

---

## 📸 Ejemplo de Layout Animado

<img src="assets/demo.gif" width="300" alt="Demo de render vertical con subtítulos dinámicos">

🔥 Detección inteligente por picos de hype (Audio + Chat)  
🎧 Transcripción word-by-word con Whisper  
📱 Layout vertical dinámico adaptable (`twitch` o `center`)  
🎨 Subtítulos estilo MrBeast/TikTok con tracking activo  
🧠 Copies y títulos generados por Gemini 2.5 Flash  
🚀 Auto-upload nativo en cascada a redes sociales  

---

## 💡 Flujo de Trabajo Automatizado

1. **`worker.py` (Orquestador):** Corre de fondo monitoreando tu canal de Twitch. En el segundo en que pasas de `live` a `offline`, despierta al pipeline central pasándole el `vod_id` más reciente.
2. **`main.py` (Procesamiento Lean):**
   - Descarga únicamente el JSON del chat y el flujo de **solo audio** del stream (ahorrando un 95% de ancho de banda y almacenamiento).
   - Analiza la fusión de señales de volumen (RMS), keywords y actividad del chat para calcular el *Signal Scoring Matrix*.
   - Descarga **quirúrgicamente** mediante `yt-dlp` solo los fragmentos mp4 de 30-45 segundos de los picos más virales.
   - Renderiza con aceleración por hardware (`h264_nvenc`) el layout vertical e inyecta los subtítulos dinámicos y la miniatura (`preview.jpg`).
3. **`publisher.py` (Despliegue Multimodal):**
   - Envía el fragmento de texto y la miniatura a la API de **Gemini** para estructurar un payload con títulos enganchadores y tags optimizados para SEO.
   - Distribuye en cascada el clip final a tus cuentas de **YouTube Shorts, TikTok e Instagram Reels**.

---

## ⚡ Features

- 🧠 **Detección inteligente de clips:** Multi-signal scoring balanceado (Audio + Chat + Texto).
- 🎧 **Transcripción nativa:** Procesamiento con `faster-whisper` (GPU CUDA opcional).
- 💬 **Análisis de chat de Twitch:** Ponderación extra si el mensaje viene de Moderadores, VIPs o el propio Streamer.
- 📱 **Layout vertical adaptable:**
  - `twitch`: (Facecam superior fija + Gameplay central expandido).
  - `center`: (Crop centrado 9:16 ideal para vlogs o contenido de YouTube).
- 🎨 **Subtítulos estilo viral:** Animación word-by-word highlight con estilos ASS personalizables.
- ⚡ **Infraestructura elástica:** Descargas parciales ultra rápidas y render con GPU (`h264_nvenc`) + fallback automático a CPU (`libx264`).
- 🧹 **Mantenimiento automatizado:** Limpieza de artefactos y cachés pesadas al finalizar cada pipeline.

---

## ⚠️ Restricción IMPORTANTE del layout vertical

> 🚨 **Regla obligatoria para el modo `twitch`:**

EJEMPLO DE LAYOUT  
<img src="assets/demo_horizontal.png" width="300">

- La **cámara SIEMPRE debe estar del lado superior izquierdo del video** en tu lienzo de OBS/Streamlabs.
- El layout vertical está diseñado estructuralmente como:
  - Parte superior $\rightarrow$ **Facecam** (Recorte guiado por variables de entorno)
  - Parte inferior $\rightarrow$ **Gameplay** (Con escalado de impacto reactivo al score del clip)

RESULTADO  
<img src="assets/demo_vertical.png" width="300">

### ❗ ¿Por qué es obligatorio?

El sistema automatizado asume esta configuración por defecto para:
- 📱 Respetar las *safe areas* nativas de las interfaces de TikTok, Instagram y Shorts.
- 👀 Garantizar enganche visual inmediato posicionando el rostro arriba de los subtítulos.
- 🎯 Aplicar crops por coordenadas estáticas (`VERTICAL_FACE_X`, etc.) sin romper la composición general.

### 🧠 ¿Qué hacer si tu cámara está en otra posición?

#### ✅ Opción Recomendada
Si tu cámara está a la derecha, al centro o juegas sin cámara, utiliza el modo de crop centrado inteligente:
```bash
python main.py --vertical-mode center
```
---

## 📦 Requisitos

- Python **3.11**
- FFmpeg instalado en el sistema y mapeado en el PATH.
- TwitchDownloaderCLI mapeado en el PATH.
- GPU NVIDIA con drivers CUDA actualizados (Altamente recomendado para producción).

---

## 🔧 Instalación

```bash
git clone [https://github.com/LuisSotelo/vod-to-viral.git](https://github.com/LuisSotelo/vod-to-viral.git)
cd vod-to-viral

# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias core y wrappers de APIs
pip install -r requirements.txt
```

---

## 📦 requirements.txt

El proyecto usa las siguientes dependencias de Python:

```txt
# Core Pipeline
python-dotenv
requests==2.32.3
charset-normalizer==3.3.2
yt-dlp>=2024.3.10
opencv-python-headless
numpy
pandas
scipy
tqdm

# AI, Audio & Imaging
torch>=2.2.0
faster-whisper
librosa>=0.10.1
soundfile
google-genai
pillow

# Subtitles & Renders
pysrt
srt

# Platform APIs (OAuth & Services)
google-api-python-client
google-auth-oauthlib

# Performance Utilities
orjson
rich
python-dateutil
colorlog
beautifulsoup4
lxml
```

---

## ⚙️ Configuración del Entorno (.env)

Crea un archivo .env en la raíz del proyecto para inicializar las credenciales de las APIs:

```bash
# 🔐 TWITCH CREDENTIALS ([https://dev.twitch.tv/console/apps](https://dev.twitch.tv/console/apps))
TWITCH_CLIENT_ID=tu_client_id_aqui
TWITCH_CLIENT_SECRET=tu_client_secret_aqui
TWITCH_USER_LOGIN=luishongo

# 🧠 AI CONFIGURATION (Google AI Studio)
GEMINI_API_KEY=tu_api_key_de_gemini_aqui

# 🚀 MULTIPLATFORM PUBLISHING TOKENS
TIKTOK_ACCESS_TOKEN=tu_token_de_tiktok_aqui
INSTAGRAM_ACCESS_TOKEN=tu_token_de_graph_api_meta_aqui
INSTAGRAM_ACCOUNT_ID=tu_id_numerico_de_cuenta_comercial_aqui

# 📁 CARPETAS Y COORDENADAS DE RENDERING
OUTPUT_DIR=./out
VERTICAL_TOP_H=620
VERTICAL_GAP_H=24
VERTICAL_BOT_H=1276
VERTICAL_FACE_X=0
VERTICAL_FACE_Y=8
VERTICAL_FACE_W=620
VERTICAL_FACE_H=390
```

## ▶️ Modos de Uso
🧠 Despliegue de Producción (100% Autónomo)
Deja corriendo el proceso demonio para que escuche tu canal. Publicará los clips de forma desatendida en cuanto cierres stream:
```bash
python worker.py
```

🔵 Modo Manual (Twitch VOD por ID)
Procesa un stream específico de forma manual inyectando su identificador:
```bash
python main.py --vod-id 2730289595 --vertical
```

🔴 Modo Inyección Externa (YouTube Videos)
Extrae picos de interés y renderiza clips verticales de videos largos de YouTube:
```bash
python main.py --youtube-url "[https://www.youtube.com/watch?v=XXXXXX](https://www.youtube.com/watch?v=XXXXXX)" --vertical
```

👁️ Modo Calibración Visual (Preview)
Genera instantáneamente un frame de prueba (preview.jpg) para ajustar las coordenadas de tu facecam sin renderizar video:
```bash
python main.py --max-clips 1 --vertical --preview-vertical
```

---

## 📁 Estructura del proyecto

```bash
vod-to-viral/
├── worker.py          # Proceso demonio de escucha activa (Orquestador principal)
├── main.py            # Core del pipeline: Análisis analógico, cortes quirúrgicos y renders
├── publisher.py       # Capa de abstracción multiplataforma, Gemini AI Studio y envíos API
├── requirements.txt   # Manifiesto de dependencias fijadas
├── client_secrets.json# Credenciales OAuth2 locales para la API de YouTube
├── .env.example       # Plantilla de variables de entorno
└── out/               # Estructura de almacenamiento temporal indexada por VOD ID
```
---

## 🎥 Output
El sistema indexa los resultados de manera limpia dentro del directorio de salida:

🎬 Clips horizontales (YouTube)
📱 Clips verticales (TikTok / Shorts / Reels)
💬 Subtítulos sincronizados

Ejemplo:
```bash
out/[VOD_ID]/
├── [VOD_ID]_chat.json  # Descarga cruda del chat de Twitch
├── process.log         # Trazabilidad completa de renders y peticiones API
├── report.json         # Timestamps, métricas y metadatos de los clips elegidos
└── clips/
    ├── clip_85_3600-3645_16x9.mp4  # Formato horizontal master
    ├── clip_85_3600-3645_9x16.mp4  # Formato vertical final con subs quemados
    └── preview_3600-3645.jpg       # Portada/Thumbnail optimizada para feeds
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
  ```
