#!/usr/bin/env python3
"""
README
======

Este script descarga el VOD más reciente de un canal de Twitch, analiza el
contenido para generar clips automáticamente y exporta cada clip con subtítulos
y estilos atractivos.

Uso básico:
  Clips Normales
    python main.py --max-clips 5
  Clips Verticales
    python main.py --max-clips 5 --vertical
  Preview del layout vertical
    python main.py --max-clips 1 --vertical --preview-vertical
  Afinar detección
    python main.py --peak-height 0.20 --min-peak-distance-sec 5

Requisitos previos
------------------
1. Python 3.11 recomendado
2. ffmpeg instalado y disponible en PATH.
3. TwitchDownloaderCLI instalado y disponible en PATH.
4. Archivo `.env` con:
   TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, TWITCH_USER_LOGIN, [OUTPUT_DIR]

Notas
-----
- La transcripción usa faster-whisper con word timestamps reales.
- Si hay GPU CUDA disponible, Whisper usará GPU automáticamente.
- El render intentará usar NVENC; si falla, hace fallback a CPU.
- El layout vertical usa coordenadas configurables desde `.env`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> logging.Logger:
    """Configura logging estructurado en consola y archivo."""
    logger = logging.getLogger("twitch_clip")
    logger.setLevel(logging.DEBUG)

    # Evita handlers duplicados si el módulo se recarga
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Carga de entorno y argumentos
# ---------------------------------------------------------------------------


@dataclass
class Config:
    client_id: str
    client_secret: str
    user_login: str
    output_dir: Path
    max_clips: int
    min_clip_sec: int
    max_clip_sec: int
    vertical: bool
    skip_download: bool
    vod_id: Optional[str]
    model_size: str
    peak_height: float
    min_peak_distance_sec: Optional[float]
    reset: str | None
    reset_only: bool
    preview_vertical: bool
    vertical_top_h: int
    vertical_gap_h: int
    vertical_bot_h: int
    vertical_face_x: int
    vertical_face_y: int
    vertical_face_w: int
    vertical_face_h: int
    vertical_mode: str
    youtube_url: Optional[str]


def load_env_and_args() -> Config:
    """Carga variables de entorno desde .env y argumentos de CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Descarga el VOD más reciente de un canal de Twitch, analiza audio, "
            "texto, chat y movimiento, y genera clips automáticos en formato "
            "horizontal y/o vertical."
        ),
        epilog=(
            "Ejemplos de uso:\n"
            "  python main.py --max-clips 5\n"
            "  python main.py --max-clips 5 --vertical\n"
            "  python main.py --max-clips 1 --vertical --preview-vertical\n"
            "  python main.py --vod-id 2730289595 --max-clips 3 --vertical\n"
            "  python main.py --peak-height 0.20 --min-peak-distance-sec 5\n"
            "  python main.py --reset cache --reset-only\n\n"
            "Variables requeridas en .env:\n"
            "  TWITCH_CLIENT_ID\n"
            "  TWITCH_CLIENT_SECRET\n"
            "  TWITCH_USER_LOGIN\n"
            "  OUTPUT_DIR (opcional, default: ./out)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    general = parser.add_argument_group("Opciones generales")
    general.add_argument(
        "--max-clips",
        type=int,
        default=10,
        help="Número máximo de clips a generar (default: %(default)s)",
    )
    general.add_argument(
        "--min-clip-sec",
        type=int,
        default=15,
        help="Duración mínima usada como referencia para separación entre clips (default: %(default)s)",
    )
    general.add_argument(
        "--max-clip-sec",
        type=int,
        default=45,
        help="Duración máxima de cada clip en segundos (default: %(default)s)",
    )
    general.add_argument(
        "--vod-id",
        type=str,
        help="Procesa un VOD específico por ID en lugar de usar el más reciente",
    )
    general.add_argument(
        "--skip-download",
        action="store_true",
        help="No descarga el VOD; espera encontrarlo ya en OUTPUT_DIR",
    )
    general.add_argument(
        "--youtube-url",
        type=str,
        help="URL de video de YouTube para procesar",
    )

    render = parser.add_argument_group("Opciones de render")
    render.add_argument(
        "--vertical",
        action="store_true",
        help="Genera también la versión vertical 9:16",
    )
    render.add_argument(
        "--preview-vertical",
        action="store_true",
        help="Genera una imagen preview del layout vertical en lugar del video vertical",
    )
    render.add_argument(
        "--vertical-mode",
        type=str,
        choices=["twitch", "center"],
        default="twitch",
        help="Modo de render vertical: 'twitch' (facecam + gameplay) o 'center' (crop centrado 9:16)",
    )

    detection = parser.add_argument_group("Opciones de detección")
    detection.add_argument(
        "--model-size",
        type=str,
        default="medium",
        help="Modelo de Whisper a usar: tiny, base, small, medium, large (default: %(default)s)",
    )
    detection.add_argument(
        "--peak-height",
        type=float,
        default=0.5,
        help="Umbral de altura para detectar picos en la señal fusionada [0..1] (default: %(default)s)",
    )
    detection.add_argument(
        "--min-peak-distance-sec",
        type=float,
        default=None,
        help="Distancia mínima entre picos en segundos. Si no se define, usa --min-clip-sec",
    )

    maintenance = parser.add_argument_group("Opciones de mantenimiento")
    maintenance.add_argument(
        "--reset",
        nargs="?",
        const="cache",
        choices=["cache", "all"],
        help=(
            "Limpia temporales antes de ejecutar.\n"
            "  cache -> borra artefactos y conserva VODs\n"
            "  all   -> borra también videos descargados"
        ),
    )
    maintenance.add_argument(
        "--reset-only",
        action="store_true",
        help="Limpia según --reset y termina sin ejecutar el pipeline",
    )

    args = parser.parse_args()

    if args.preview_vertical and not args.vertical:
        parser.error("--preview-vertical requiere --vertical")

    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")
    user_login = os.getenv("TWITCH_USER_LOGIN")
    output_dir = Path(os.getenv("OUTPUT_DIR", "./out"))

    if not args.youtube_url:
        if not client_id or not client_secret or not user_login:
            raise ValueError("Faltan variables obligatorias de Twitch en .env")

    return Config(
        client_id=client_id,
        client_secret=client_secret,
        user_login=user_login,
        output_dir=output_dir,
        max_clips=args.max_clips,
        min_clip_sec=args.min_clip_sec,
        max_clip_sec=args.max_clip_sec,
        vertical=args.vertical,
        skip_download=args.skip_download,
        vod_id=args.vod_id,
        model_size=args.model_size,
        peak_height=args.peak_height,
        min_peak_distance_sec=args.min_peak_distance_sec,
        reset=args.reset,
        reset_only=args.reset_only,
        preview_vertical=args.preview_vertical,
        vertical_top_h=int(os.getenv("VERTICAL_TOP_H", "620")),
        vertical_gap_h=int(os.getenv("VERTICAL_GAP_H", "24")),
        vertical_bot_h=int(os.getenv("VERTICAL_BOT_H", "1276")),
        vertical_face_x=int(os.getenv("VERTICAL_FACE_X", "0")),
        vertical_face_y=int(os.getenv("VERTICAL_FACE_Y", "8")),
        vertical_face_w=int(os.getenv("VERTICAL_FACE_W", "620")),
        vertical_face_h=int(os.getenv("VERTICAL_FACE_H", "390")),
        youtube_url=args.youtube_url,
        vertical_mode=args.vertical_mode,
    )

def get_nvenc_quality_args() -> list[str]:
    return [
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-cq", "18",
        "-rc", "vbr",
        "-b:v", "0",
        "-maxrate", "25M",
        "-bufsize", "50M",
        "-pix_fmt", "yuv420p",
    ]


def get_x264_quality_args() -> list[str]:
    return [
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
    ]

# ---------------------------------------------------------------------------
# Funciones de API Twitch
# ---------------------------------------------------------------------------


class TwitchAPIError(RuntimeError):
    """Error al interactuar con la API de Twitch."""


def get_access_token(cfg: Config, logger: logging.Logger) -> str:
    """Siempre obtiene un token nuevo (App Access Token)."""
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "grant_type": "client_credentials",
    }

    logger.info("Solicitando App Access Token a Twitch")
    resp = requests.post(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise TwitchAPIError(f"Error {resp.status_code} al obtener token: {resp.text}")

    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise TwitchAPIError("Respuesta inválida al obtener token")

    logger.info("Token generado correctamente")
    return token


def helix_get(
    path: str, cfg: Config, token: str, params: Optional[Dict[str, str]] = None
) -> Dict:
    url = f"https://api.twitch.tv/helix/{path}"
    headers = {
        "Client-ID": cfg.client_id,
        "Authorization": f"Bearer {token}",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code == 401:
        raise TwitchAPIError(f"Error 401 en Helix: {resp.text}")
    if resp.status_code in {403, 404}:
        raise TwitchAPIError(f"Error {resp.status_code} en Helix: {resp.text}")

    resp.raise_for_status()
    return resp.json()


def get_user_id(cfg: Config, token: str, logger: logging.Logger) -> str:
    data = helix_get("users", cfg, token, params={"login": cfg.user_login})
    users = data.get("data", [])
    if not users:
        raise TwitchAPIError("Usuario de Twitch no encontrado")
    user_id = users[0]["id"]
    logger.info("user_id encontrado: %s", user_id)
    return user_id


def get_latest_vod(
    user_id: str, cfg: Config, token: str, logger: logging.Logger
) -> Dict:
    params = {"user_id": user_id, "type": "archive", "first": 1}
    data = helix_get("videos", cfg, token, params=params)
    videos = data.get("data", [])
    if not videos:
        raise TwitchAPIError("No hay VOD disponible")
    vod = videos[0]
    logger.info("Último VOD: %s", vod.get("id"))
    return vod


# ---------------------------------------------------------------------------
# Descarga de VOD y chat
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in " -_").rstrip()


def download_vod_if_needed(vod: Dict, cfg: Config, logger: logging.Logger) -> Path:
    """Descarga el VOD usando yt-dlp si no existe."""
    from yt_dlp import YoutubeDL  # import tardío

    vod_id = vod["id"]
    title = sanitize_filename(vod.get("title", "vod"))
    created_at = vod.get("created_at", "").split("T")[0]
    filename = f"{vod_id}_{created_at}_{title}.mp4"

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = cfg.output_dir / filename
    archive = cfg.output_dir / "download_archive.txt"

    if video_path.exists():
        logger.info("VOD ya existe en disco: %s", video_path)
        return video_path

    if cfg.skip_download:
        raise FileNotFoundError("Se omitió la descarga pero el VOD no está disponible")

    logger.info("Descargando VOD %s", vod_id)
    ydl_opts = {
        "outtmpl": str(video_path),
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "continuedl": True,
        "overwrites": False,
        "download_archive": str(archive),
        "consoletitle": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([vod["url"]])

    return video_path


def download_chat_via_twitchdownloadercli(
    vod_id: int | str, out_json: Path, logger: logging.Logger
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "TwitchDownloaderCLI",
        "chatdownload",
        "-u",
        str(vod_id),
        "-o",
        str(out_json),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        logger.error(
            "TwitchDownloaderCLI falló (%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
            res.returncode,
            res.stdout,
            res.stderr,
        )
        raise RuntimeError(f"TwitchDownloaderCLI returned {res.returncode}")


def download_chat(vod_id, cfg, logger, chat_path: Path) -> bool:
    """Descarga el chat o continúa sin chat si falla."""
    if chat_path.exists() and chat_path.stat().st_size > 0:
        logger.info("Chat ya existe: %s (skip)", chat_path)
        return True

    logger.info("Descargando chat con TwitchDownloaderCLI...")
    try:
        download_chat_via_twitchdownloadercli(vod_id, chat_path, logger)
        if chat_path.exists() and chat_path.stat().st_size > 0:
            logger.info("Chat descargado correctamente: %s", chat_path)
            return True
        logger.warning("TwitchDownloaderCLI terminó pero no dejó chat utilizable.")
        return False
    except Exception as e:
        logger.warning("No se pudo descargar el chat. Continuaré sin chat. Motivo: %s", e)
        return False


# ---------------------------------------------------------------------------
# Descarga de VOD y chat
# ---------------------------------------------------------------------------

def download_youtube_video(url: str, cfg: Config, logger: logging.Logger) -> Path:
    """Descarga un video de YouTube usando yt-dlp en la mejor calidad posible."""
    from yt_dlp import YoutubeDL

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(cfg.output_dir / "%(id)s_%(title).80s.%(ext)s"),
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
    }

    logger.info("Descargando video de YouTube en máxima calidad...")

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        ext = info.get("ext", "")
        video_path = Path(filename)

        # Si yt-dlp fusionó a mp4, el nombre final puede cambiar
        if ext and video_path.suffix.lower() != ".mp4":
            candidate_mp4 = video_path.with_suffix(".mp4")
            if candidate_mp4.exists():
                video_path = candidate_mp4

    if not video_path.exists():
        raise FileNotFoundError("No se pudo descargar el video de YouTube")

    logger.info("Video descargado: %s", video_path)
    return video_path

# ---------------------------------------------------------------------------
# Análisis de video y audio
# ---------------------------------------------------------------------------


def analyze_motion(video_path: Path, logger: logging.Logger, fps: int = 2) -> pd.DataFrame:
    """Calcula magnitud de movimiento muestreando el video."""
    import cv2

    cache_path = video_path.with_suffix(".motion.csv")
    if cache_path.exists():
        try:
            logger.info("Cargando movimiento desde cache: %s", cache_path)
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning("No se pudo leer cache de movimiento (%s), recalculando.", e)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video para analizar movimiento")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_interval = max(1, int(video_fps // fps)) if video_fps > 0 else 1

    prev_gray = None
    idx = 0
    rows = []

    pbar = tqdm(
        desc="Movimiento",
        unit="frm",
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = float(np.mean(diff))
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                rows.append({"time": t, "motion": motion})
            prev_gray = gray

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    df = pd.DataFrame(rows)
    try:
        df.to_csv(cache_path, index=False)
        logger.info("Movimiento guardado en cache: %s", cache_path)
    except Exception as e:
        logger.warning("No se pudo guardar cache de movimiento: %s", e)

    return df


def extract_audio(video_path: Path, logger: logging.Logger) -> Path:
    """Extrae audio a WAV mono 16 kHz."""
    wav_path = video_path.with_suffix(".wav")
    if wav_path.exists():
        logger.debug("Audio ya extraído: %s", wav_path)
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    logger.info("Extrayendo audio")
    subprocess.run(cmd, check=True)
    return wav_path


def analyze_audio_emotion(audio_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Calcula una señal de emoción positiva simple."""
    import librosa

    rows = []

    try:
        import torch

        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
            force_reload=False,
        )

        if isinstance(utils, dict):
            get_speech_timestamps = utils.get("get_speech_timestamps")
        elif isinstance(utils, (list, tuple)) and len(utils) > 0 and callable(utils[0]):
            get_speech_timestamps = utils[0]
        else:
            get_speech_timestamps = None

        if get_speech_timestamps is None:
            raise RuntimeError("Silero utils no contiene get_speech_timestamps")

        wav, sr = librosa.load(str(audio_path), sr=16000)
        _ = get_speech_timestamps(torch.from_numpy(wav), model, sampling_rate=sr)  # type: ignore[arg-type]

        rms = librosa.feature.rms(y=wav, frame_length=sr // 2, hop_length=sr // 2)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=sr // 2)
        for t, e in zip(times, rms):
            rows.append({"time": float(t), "audio_score": float(e)})

        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("Silero VAD no disponible (%s). Uso heurísticas con librosa.", e)

    wav, sr = librosa.load(str(audio_path), sr=16000)
    rms = librosa.feature.rms(y=wav, frame_length=sr // 2, hop_length=sr // 2)[0]
    zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=sr // 2, hop_length=sr // 2)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=sr // 2)

    for t, e, z in zip(times, rms, zcr):
        score = float(e + 0.2 * z)
        rows.append({"time": float(t), "audio_score": score})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transcripción ASR
# ---------------------------------------------------------------------------


def _safe_load_segments_json(path: Path, logger: logging.Logger) -> list[dict] | None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        txt = path.read_text(encoding="utf-8", errors="strict")
        return json.loads(txt)
    except Exception as e:
        logger.warning("segments.json inválido (%s). Reharé transcripción.", e)
        return None


def _atomic_write_json(path: Path, data: dict | list, logger: logging.Logger) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def get_torch_device_and_compute_type(logger: logging.Logger) -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA disponible. Usando GPU: %s", gpu_name)
            return "cuda", "float16"

        logger.info("CUDA no disponible. Usando CPU.")
        return "cpu", "float32"

    except Exception as e:
        logger.warning("No se pudo inicializar torch CUDA (%s). Usando CPU.", e)
        return "cpu", "float32"


def get_ffmpeg_video_encoder() -> str:
    return "h264_nvenc"


def ffmpeg_hwaccel_input_args() -> list[str]:
    return ["-hwaccel", "cuda"]


def transcribe(
    video_path: Path, cfg: Config, logger: logging.Logger
) -> list[dict]:
    from faster_whisper import WhisperModel

    seg_path = video_path.with_suffix(".segments.json")

    segs = _safe_load_segments_json(seg_path, logger)
    if segs is not None:
        logger.info("Segments existentes: %s", seg_path)
        return segs

    device, compute_type = get_torch_device_and_compute_type(logger)
    logger.info(
        "Transcribiendo audio con modelo %s | device=%s | compute_type=%s",
        cfg.model_size,
        device,
        compute_type,
    )

    model = WhisperModel(
        cfg.model_size,
        device=device,
        compute_type=compute_type,
    )

    segments, _info = model.transcribe(
        str(video_path),
        language="es",
        task="transcribe",
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=True,
        initial_prompt=(
            "El audio es en español de México. Usa puntuación natural "
            "y conserva jerga gamer/stream. No traduzcas."
        ),
    )

    rows: list[dict] = []

    for seg in segments:
        start = float(seg.start)
        end = float(seg.end)
        text = (seg.text or "").strip()

        words: list[dict] = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                w_word = (getattr(w, "word", "") or "").strip()
                w_start = getattr(w, "start", None)
                w_end = getattr(w, "end", None)

                if not w_word or w_start is None or w_end is None:
                    continue

                words.append(
                    {
                        "word": w_word,
                        "start": float(w_start),
                        "end": float(w_end),
                    }
                )

        rows.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "confidence": float(getattr(seg, "avg_logprob", 0.0)),
                "words": words,
            }
        )

    _atomic_write_json(seg_path, rows, logger)
    logger.info("Segments guardados en: %s", seg_path)
    return rows

def ass_escape_text(text: str) -> str:
    return (
        text.replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def chunk_words_for_display(words: list[dict], max_words: int = 4) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []

    for w in words:
        word_txt = (w.get("word") or "").strip()
        if not word_txt:
            continue

        current.append(w)

        if len(current) >= max_words:
            chunks.append(current)
            current = []
        elif word_txt.endswith((".", "?", "!", ",")):
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    return chunks


def ffmpeg_ass_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", "\\:")


def make_retro_ass_from_segments(
    clip_segments: list[dict],
    clip_ass: Path,
    clip_start: float,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    max_words: int = 4,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> bool:
    events: list[tuple[float, float, str]] = []

    def split_chunk_into_lines(chunk: list[dict]) -> list[list[dict]]:
        """
        Divide un chunk en líneas según límite de caracteres aproximado.
        Máximo max_lines líneas; si excede, compacta la última.
        """
        lines: list[list[dict]] = []
        current: list[dict] = []
        current_len = 0

        for word_info in chunk:
            word = str(word_info["word"]).strip()
            add_len = len(word) if not current else len(word) + 1

            if current and (current_len + add_len) > max_chars_per_line:
                lines.append(current)
                current = [word_info]
                current_len = len(word)
            else:
                current.append(word_info)
                current_len += add_len

        if current:
            lines.append(current)

        if len(lines) <= max_lines:
            return lines

        # Si salen demasiadas líneas, colapsa todo lo sobrante en la última permitida
        merged = lines[: max_lines - 1]
        tail: list[dict] = []
        for extra_line in lines[max_lines - 1 :]:
            tail.extend(extra_line)
        merged.append(tail)
        return merged

    for seg in clip_segments:
        seg_words = seg.get("words") or []
        if not seg_words:
            continue

        valid_words = []
        for w in seg_words:
            w_word = (w.get("word") or "").strip()
            w_start = w.get("start")
            w_end = w.get("end")
            if not w_word or w_start is None or w_end is None:
                continue

            valid_words.append({
                "word": w_word,
                "start": float(w_start) - clip_start,
                "end": float(w_end) - clip_start,
            })

        if not valid_words:
            continue

        chunks = chunk_words_for_display(valid_words, max_words=max_words)

        for chunk in chunks:
            if not chunk:
                continue

            line_groups = split_chunk_into_lines(chunk)

            chunk_start = max(0.0, float(chunk[0]["start"]))
            chunk_end = max(chunk_start + 0.05, float(chunk[-1]["end"]) + 0.10)

            for active_idx, active_word in enumerate(chunk):
                ev_start = max(0.0, float(active_word["start"]))

                if active_idx + 1 < len(chunk):
                    ev_end = max(ev_start + 0.02, float(chunk[active_idx + 1]["start"]))
                else:
                    ev_end = chunk_end

                styled_lines: list[str] = []
                running_idx = 0

                for line in line_groups:
                    styled_words = []
                    for word_info in line:
                        safe_word = ass_escape_text(str(word_info["word"]).upper())

                        if running_idx == active_idx:
                            styled_word = r"{\c&H00FFFF&\bord8\shad2}" + safe_word + r"{\r}"
                        else:
                            styled_word = r"{\c&HFFFFFF&\bord7\shad2}" + safe_word + r"{\r}"

                        styled_words.append(styled_word)
                        running_idx += 1

                    styled_lines.append(" ".join(styled_words))

                full_text = r"\N".join(styled_lines)
                events.append((ev_start, ev_end, full_text))

    def fmt_ass_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Retro,Press Start 2P,58,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    for start, end, text in events:
        line = r"{\fad(20,20)}" + text
        ass += f"Dialogue: 0,{fmt_ass_time(start)},{fmt_ass_time(end)},Retro,,0,0,0,,{line}\n"

    clip_ass.write_text(ass, encoding="utf-8-sig")
    return len(events) > 0

# ---------------------------------------------------------------------------
# Análisis de texto y chat
# ---------------------------------------------------------------------------


KEYWORDS_TEXT = {
    "jajaja",
    "jaja",
    "risa",
    "wtf",
    "no mames",
    "vamos",
    "épico",
    "gg",
    "lol",
    "xd",
    "nooo",
    "boom",
    "epico",
    "increíble",
    "clip",
    "clipeen",
    "culo",
    "huevo",
    "ayuda",
    "pendejo",
    "verga",
    "morra",
    "morrita",
}

KEYWORDS_CHAT = {
    "jajaja",
    "xd",
    "clip",
    "no mames",
    "épico",
    "wtf",
    "gg",
    "jaja",
    "noooo",
    "lol",
    "ooooh",
    "pov",
    "like",
    "f",
    "ole",
    "vamo",
    "crack",
    "eso",
}


def _extract_chat_messages(chat_json: dict | list):
    """Devuelve (offset_segundos, texto, meta) desde el JSON del chat."""

    def _join_fragments(msg: dict) -> str:
        frags = msg.get("fragments")
        if isinstance(frags, list) and frags:
            parts = []
            for f in frags:
                t = f.get("text") or ""
                emo = f.get("emoticon")
                if emo and isinstance(emo, dict):
                    t = t or f":{emo.get('emoticon_id', 'emote')}:"
                parts.append(str(t))
            return "".join(parts) or str(msg.get("body") or "")
        return str(msg.get("body") or "")

    def _emit(obj):
        try:
            t = float(obj.get("content_offset_seconds", 0.0) or 0.0)
        except Exception:
            t = 0.0

        msg = obj.get("message") or {}
        text = _join_fragments(msg) if isinstance(msg, dict) else (str(msg) if isinstance(msg, str) else "")
        commenter = obj.get("commenter") or {}

        badges = set()
        for b in (msg.get("user_badges") or []):
            bid = str(b.get("_id") or "").lower()
            if bid:
                badges.add(bid)

        meta = {
            "badges": badges,
            "is_streamer": str(commenter.get("name") or "").lower()
            == str((chat_json.get("streamer") or {}).get("login") or "").lower(),
        }

        if text:
            return (t, text, meta)
        return None

    if isinstance(chat_json, dict) and isinstance(chat_json.get("comments"), list):
        for item in chat_json["comments"]:
            if isinstance(item, dict):
                pair = _emit(item)
                if pair:
                    yield pair
        return

    if isinstance(chat_json, dict):
        for key in ("data",):
            seq = chat_json.get(key)
            if isinstance(seq, list):
                for item in seq:
                    if isinstance(item, dict):
                        pair = _emit(item)
                        if pair:
                            yield pair
    elif isinstance(chat_json, list):
        for item in chat_json:
            if isinstance(item, dict):
                pair = _emit(item)
                if pair:
                    yield pair


def score_text_and_chat(segments, chat_path, logger):
    rows_text = []
    for seg in segments:
        stext = str(seg.get("text", "")).lower()
        if any(kw in stext for kw in KEYWORDS_TEXT):
            rows_text.append({"time": float(seg.get("start", 0.0)), "text_score": 1.0})
    df_text = pd.DataFrame(rows_text)

    rows_chat = []
    if chat_path.exists() and chat_path.stat().st_size > 0:
        try:
            chat_data = json.loads(chat_path.read_text(encoding="utf-8"))
            for t, text, meta in _extract_chat_messages(chat_data):
                ltext = text.lower()
                base = sum(1.0 for kw in KEYWORDS_CHAT if kw in ltext)
                if base <= 0:
                    continue

                weight = 1.0
                if meta.get("is_streamer"):
                    weight += 0.5

                badges = meta.get("badges") or set()
                if "vip" in badges:
                    weight += 0.25
                if "moderator" in badges:
                    weight += 0.25
                if "subscriber" in badges:
                    weight += 0.10

                rows_chat.append({"time": float(t), "chat_score": float(base * weight)})

        except Exception as e:
            logger.warning("No se pudo parsear chat.json (%s), continúo sin chat.", e)
    else:
        logger.info("No hay chat disponible para este VOD; se omitirá la señal de chat.")

    df_chat = pd.DataFrame(rows_chat)
    return df_text, df_chat


# ---------------------------------------------------------------------------
# Fusión de señales y selección de clips
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    start: float
    end: float
    score: float


def merge_signals_and_pick_clips(
    motion: pd.DataFrame,
    audio: pd.DataFrame,
    text: pd.DataFrame,
    chat: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
) -> List[Clip]:
    """Combina señales normalizadas y selecciona los mejores clips."""

    def to_1s(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df is None or df.empty or "time" not in df.columns:
            return pd.DataFrame(columns=["time", col])

        g = df.copy()
        g["sec"] = g["time"].round().astype(int)
        g = g.groupby("sec")[col].max().reset_index().rename(columns={"sec": "time"})
        return g

    max_time = 0.0
    for d in (motion, audio, text, chat):
        if d is not None and not d.empty:
            mt = float(d["time"].max())
            if mt > max_time:
                max_time = mt

    timeline = pd.DataFrame({"time": np.arange(0, int(max_time) + 1)})

    m1 = to_1s(motion, "motion")
    a1 = to_1s(audio, "audio_score")
    t1 = to_1s(text, "text_score")
    c1 = to_1s(chat, "chat_score")

    df = (
        timeline.merge(m1, on="time", how="left")
        .merge(a1, on="time", how="left")
        .merge(t1, on="time", how="left")
        .merge(c1, on="time", how="left")
        .fillna(0.0)
    )

    for col in ["motion", "audio_score", "text_score", "chat_score"]:
        m = float(df[col].max())
        if m > 0:
            df[col] = df[col] / m

    if cfg.youtube_url:
        df["merged"] = (
            df["audio_score"] * 0.65
            + df["motion"] * 0.35
        )
    else:
        df["merged"] = (
            df["audio_score"] * 0.35
            + df["text_score"] * 0.30
            + df["chat_score"] * 0.25
            + df["motion"] * 0.10
        )

    df["smooth"] = df["merged"].rolling(window=5, min_periods=1, center=True).max()

    from scipy.signal import find_peaks

    height = cfg.peak_height

    time_arr = df["time"].to_numpy()
    if len(time_arr) >= 2:
        dt = np.median(np.diff(time_arr))
        dt = float(dt) if dt and dt > 0 else 1.0
    else:
        dt = 1.0

    min_dist_sec = cfg.min_peak_distance_sec
    if min_dist_sec is None:
        min_dist_sec = float(cfg.min_clip_sec)

    distance = max(1, int(round(min_dist_sec / dt)))
    peaks, _props = find_peaks(df["smooth"].to_numpy(), height=height, distance=distance)

    candidates: List[Clip] = []
    for p in peaks:
        t = float(df.loc[p, "time"])
        start = max(0.0, t - max(cfg.min_clip_sec / 2, 6))
        end = start + float(cfg.max_clip_sec)
        score = float(df.loc[p, "smooth"])
        candidates.append(Clip(start=start, end=end, score=score))

    def iou(a: Clip, b: Clip) -> float:
        inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
        union = (a.end - a.start) + (b.end - b.start) - inter
        return inter / union if union > 0 else 0.0

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected: List[Clip] = []
    for c in candidates:
        if all(iou(c, s) < 0.4 for s in selected):
            selected.append(c)
        if len(selected) >= cfg.max_clips:
            break

    logger.info("Clips seleccionados: %d", len(selected))
    return selected


# ---------------------------------------------------------------------------
# Renderizado de clips
# ---------------------------------------------------------------------------

def extract_segment_words_for_clip(
    segments: list[dict],
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    clip_segments: list[dict] = []

    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))

        if seg_end < clip_start or seg_start > clip_end:
            continue

        words = []
        for w in seg.get("words", []) or []:
            w_start = w.get("start")
            w_end = w.get("end")
            w_word = (w.get("word") or "").strip()

            if w_start is None or w_end is None or not w_word:
                continue

            w_start = float(w_start)
            w_end = float(w_end)

            if w_end < clip_start or w_start > clip_end:
                continue

            words.append({"word": w_word, "start": w_start, "end": w_end})

        clip_segments.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": seg.get("text", ""),
                "words": words,
            }
        )

    return clip_segments


def render_clip_with_subs(
    clip: Clip,
    video_path: Path,
    segments: list[dict],
    out_dir: Path,
    cfg: Config,
    logger: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"clip_{int(clip.score*100)}_{int(clip.start)}-{int(clip.end)}"

    clip_ass = out_dir / f"{base_name}.ass"
    clip_segments = extract_segment_words_for_clip(segments, clip.start, clip.end)
    ass_ok = make_retro_ass_from_segments(
        clip_segments,
        clip_ass,
        clip_start=clip.start,
        max_words=4,
    )

    v_in = str(video_path.resolve())
    vcodec = get_ffmpeg_video_encoder()

    # -------------------------------------------------
    # 1) Render horizontal 16:9
    # -------------------------------------------------
    out_169 = out_dir / f"{base_name}_16x9.mp4"
    cmd_cut_169 = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_hwaccel_input_args(),
        "-ss",
        f"{clip.start}",
        "-to",
        f"{clip.end}",
        "-i",
        v_in,
        *get_nvenc_quality_args(),
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-reset_timestamps",
        "1",
        "-avoid_negative_ts",
        "make_zero",
        str(out_169),
    ]
    logger.info("Render 16:9 GPU → %s", out_169)
    res = subprocess.run(
        cmd_cut_169,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        logger.warning("NVENC 16:9 falló, haré fallback CPU.\n%s", res.stderr)
        cmd_cut_169_cpu = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{clip.start}",
            "-to",
            f"{clip.end}",
            "-i",
            v_in,
            *get_x264_quality_args(),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-reset_timestamps",
            "1",
            "-avoid_negative_ts",
            "make_zero",
            str(out_169),
        ]
        res_cpu = subprocess.run(
            cmd_cut_169_cpu,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res_cpu.returncode != 0:
            logger.error("ffmpeg cut 16:9 falló.\n%s", res_cpu.stderr)
            raise RuntimeError("ffmpeg cut 16:9 failed")

    # -------------------------------------------------
    # 2) Render vertical 9:16
    # -------------------------------------------------
    if not cfg.vertical:
        return

    out = out_dir / f"{base_name}_9x16.mp4"

    if cfg.vertical_mode == "center":
        center_crop = (
            "crop="
            "ih*(9/16):"
            "ih:"
            "(iw-ih*(9/16))/2:"
            "0"
        )

        if ass_ok and clip_ass.exists() and clip_ass.stat().st_size > 0:
            ass_path = ffmpeg_ass_path(clip_ass)
            filtergraph = (
                f"[0:v]{center_crop},"
                f"scale=1080:1920:flags=lanczos,"
                f"unsharp=5:5:0.6:5:5:0.0,"
                f"setsar=1[stack];"
                f"[stack]ass='{ass_path}'[outv]"
            )
        else:
            logger.warning("No se pudo generar ASS válido. Renderizaré vertical sin subtítulos quemados.")
            filtergraph = (
                f"[0:v]{center_crop},"
                f"scale=1080:1920:flags=lanczos,"
                f"unsharp=5:5:0.6:5:5:0.0,"
                f"setsar=1[outv]"
            )

    else:
        face = (
            f"crop={cfg.vertical_face_w}:{cfg.vertical_face_h}:"
            f"{cfg.vertical_face_x}:{cfg.vertical_face_y}"
        )

        if clip.score >= 0.85:
            hype_zoom = 1.16
        elif clip.score >= 0.70:
            hype_zoom = 1.11
        else:
            hype_zoom = 1.06

        game = (
            f"crop="
            f"ih*(1080/{cfg.vertical_bot_h})/{hype_zoom}:"
            f"ih/{hype_zoom}:"
            f"(iw-ih*(1080/{cfg.vertical_bot_h})/{hype_zoom})/2:"
            f"(ih-ih/{hype_zoom})/2"
        )

        filtergraph_base = (
            f"[0:v]{face},"
            f"scale=1080:{cfg.vertical_top_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"setsar=1[face_scaled];"
            f"[face_scaled]pad=1080:{cfg.vertical_top_h}:(ow-iw)/2:(oh-ih)/2:black[face_pad];"
            f"[face_pad]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=3[face];"
            f"color=c=black:s=1080x{cfg.vertical_gap_h}:d=1[gap];"
            f"[0:v]{game},"
            f"scale=1080:{cfg.vertical_bot_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"setsar=1[game_scaled];"
            f"[game_scaled]crop=1080:{cfg.vertical_bot_h}:(iw-1080)/2:(ih-{cfg.vertical_bot_h})/2[game_final];"
            f"[face][gap][game_final]vstack=inputs=3[stack]"
        )

        if ass_ok and clip_ass.exists() and clip_ass.stat().st_size > 0:
            ass_path = ffmpeg_ass_path(clip_ass)
            filtergraph = filtergraph_base + f";[stack]ass='{ass_path}'[outv]"
        else:
            logger.warning("No se pudo generar ASS válido. Renderizaré vertical sin subtítulos quemados.")
            filtergraph = filtergraph_base + ";[stack]null[outv]"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_hwaccel_input_args(),
        "-ss",
        f"{clip.start}",
        "-to",
        f"{clip.end}",
        "-i",
        v_in,
        "-filter_complex",
        filtergraph,
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        *get_nvenc_quality_args(),
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-reset_timestamps",
        "1",
        "-avoid_negative_ts",
        "make_zero",
        str(out),
    ]

    logger.info("Render 9:16 GPU con subtítulos + punch-in → %s", out)
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        logger.warning("NVENC 9:16 falló, haré fallback CPU.\n%s", res.stderr)
        cmd_cpu = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{clip.start}",
            "-to",
            f"{clip.end}",
            "-i",
            v_in,
            "-filter_complex",
            filtergraph,
            "-map",
            "[outv]",
            "-map",
            "0:a?",
            *get_x264_quality_args(),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-reset_timestamps",
            "1",
            "-avoid_negative_ts",
            "make_zero",
            str(out),
        ]
        res_cpu = subprocess.run(
            cmd_cpu,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res_cpu.returncode != 0:
            logger.error("ffmpeg error:\n%s", res_cpu.stderr)
            raise RuntimeError("ffmpeg reaction render failed")
        
def quick_test(
    video_path: Path,
    segments: list[dict],
    cfg: Config,
    logger: logging.Logger,
) -> None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    mid_start = duration / 2
    clip = Clip(start=mid_start, end=min(duration, mid_start + 60), score=0.0)

    render_clip_with_subs(
        clip,
        video_path,
        segments,
        cfg.output_dir / "test_clip",
        cfg,
        logger,
    )

def render_vertical_preview_frame(
    clip: Clip, video_path: Path, out_dir: Path, cfg: Config, logger: logging.Logger
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"preview_{int(clip.start)}-{int(clip.end)}.jpg"

    face = (
        f"crop={cfg.vertical_face_w}:{cfg.vertical_face_h}:"
        f"{cfg.vertical_face_x}:{cfg.vertical_face_y}"
    )

    if cfg.vertical_mode == "center":
            center_crop = (
                "crop="
                "ih*(9/16):"
                "ih:"
                "(iw-ih*(9/16))/2:"
                "0"
            )

            filtergraph = (
                f"[0:v]{center_crop},"
                f"scale=1080:1920:flags=lanczos,"
                f"setsar=1[outv]"
            )
    else:
        face = (
            f"crop={cfg.vertical_face_w}:{cfg.vertical_face_h}:"
            f"{cfg.vertical_face_x}:{cfg.vertical_face_y}"
        )

        hype_zoom = 1.08
        game = (
            f"crop="
            f"ih*(1080/{cfg.vertical_bot_h})/{hype_zoom}:"
            f"ih/{hype_zoom}:"
            f"(iw-ih*(1080/{cfg.vertical_bot_h})/{hype_zoom})/2:"
            f"(ih-ih/{hype_zoom})/2"
        )

        filtergraph = (
            f"[0:v]{face},"
            f"scale=1080:{cfg.vertical_top_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"setsar=1[face_scaled];"
            f"[face_scaled]pad=1080:{cfg.vertical_top_h}:(ow-iw)/2:(oh-ih)/2:black[face_pad];"
            f"[face_pad]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=3[face];"
            f"color=c=black:s=1080x{cfg.vertical_gap_h}:d=1[gap];"
            f"[0:v]{game},"
            f"scale=1080:{cfg.vertical_bot_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"setsar=1[game_scaled];"
            f"[game_scaled]crop=1080:{cfg.vertical_bot_h}:(iw-1080)/2:(ih-{cfg.vertical_bot_h})/2[game_final];"
            f"[face][gap][game_final]vstack=inputs=3[outv]"
        )

    frame_time = clip.start + 0.5

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_hwaccel_input_args(),
        "-ss",
        f"{frame_time}",
        "-i",
        str(video_path),
        "-filter_complex",
        filtergraph,
        "-map",
        "[outv]",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]

    logger.info("Generando preview vertical → %s", out)
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        logger.warning("Preview vertical con CUDA falló, haré fallback CPU.\n%s", res.stderr)

        cmd_cpu = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{frame_time}",
            "-i",
            str(video_path),
            "-filter_complex",
            filtergraph,
            "-map",
            "[outv]",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]

        res_cpu = subprocess.run(
            cmd_cpu,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res_cpu.returncode != 0:
            logger.error("ffmpeg preview error (CPU fallback):\n%s", res_cpu.stderr)
            raise RuntimeError("ffmpeg vertical preview failed")


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def write_report(vod, clips, cfg, out_dir: Path, logger):
    report = {
        "vod_id": vod.get("id"),
        "title": vod.get("title"),
        "duration": vod.get("duration"),
        "params": vars(cfg),
        "clips": [clip.__dict__ for clip in clips],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Reporte guardado en %s", path)


# ---------------------------------------------------------------------------
# Limpieza de temporales y caché
# ---------------------------------------------------------------------------


def _safe_unlink(p: Path, logger, retries: int = 3, delay: float = 0.2) -> bool:
    if not p.exists():
        return False
    for _ in range(retries):
        try:
            p.unlink(missing_ok=True)
            return True
        except PermissionError:
            try:
                p.chmod(p.stat().st_mode | stat.S_IWUSR)
            except Exception:
                pass
            time.sleep(delay)
        except Exception as e:
            logger.debug("No pude borrar archivo %s: %s", p, e)
            return False
    return False


def _on_rmtree_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWUSR)
        func(path)
    except Exception:
        pass


def _safe_rmtree(p: Path, logger) -> bool:
    if not p.exists():
        return False
    try:
        shutil.rmtree(p, onerror=_on_rmtree_error)
        return True
    except Exception as e:
        logger.debug("No pude borrar carpeta %s: %s", p, e)
        return False


def _rm_globs(root: Path, patterns: list[str], logger) -> tuple[int, int]:
    files_deleted = 0
    dirs_deleted = 0

    for pattern in patterns:
        for path in root.rglob("*"):
            try:
                if fnmatch.fnmatch(path.name, pattern):
                    if path.is_dir():
                        if _safe_rmtree(path, logger):
                            dirs_deleted += 1
                    else:
                        if _safe_unlink(path, logger):
                            files_deleted += 1
            except Exception:
                continue

    return files_deleted, dirs_deleted


def clean_workspace(cfg, logger, mode: str = "cache") -> dict:
    """
    mode: "cache" -> borra artefactos, conserva VODs
          "all"   -> borra TODO incluido VODs
    """
    base = Path(__file__).resolve().parent
    out = cfg.output_dir

    summary = {"files": 0, "dirs": 0, "roots_removed": []}
    logger.info("Limpieza iniciada (modo=%s). output_dir=%s", mode, out)

    artifact_patterns = [
        "*.ass",
        "*.segments.json",
        "*.motion.csv",
        "*.csv",
        "*.wav",
        "*_chat.json",
        "*.html",
        "*.log",
        ".cache",
        "__pycache__",
        ".pytest_cache",
        "clips",
        "reports",
        "frames",
    ]

    f, d = _rm_globs(out, artifact_patterns, logger)
    summary["files"] += f
    summary["dirs"] += d

    if mode == "all":
        video_patterns = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.flv"]
        f2, d2 = _rm_globs(out, video_patterns, logger)
        summary["files"] += f2
        summary["dirs"] += d2

    for child in out.iterdir() if out.exists() else []:
        if child.is_dir() and child.name.lower() in {"clips", "reports", "frames"}:
            if _safe_rmtree(child, logger):
                summary["dirs"] += 1
                summary["roots_removed"].append(str(child))

    root_cache_patterns = ["__pycache__", ".cache", "pip-wheel-metadata"]
    f3, d3 = _rm_globs(base, root_cache_patterns, logger)
    summary["files"] += f3
    summary["dirs"] += d3

    try:
        ytdlp_cache = Path.home() / ".cache" / "yt-dlp"
        if ytdlp_cache.exists():
            if _safe_rmtree(ytdlp_cache, logger):
                summary["dirs"] += 1
                summary["roots_removed"].append(str(ytdlp_cache))
    except Exception:
        pass

    archive_file = base / "download_archive.txt"
    if archive_file.exists():
        if _safe_unlink(archive_file, logger):
            summary["files"] += 1
            summary["roots_removed"].append(str(archive_file))

    logger.info(
        "Limpieza terminada: %s archivos, %s carpetas eliminadas.",
        summary["files"],
        summary["dirs"],
    )
    return summary


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_env_and_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(cfg.output_dir / "process.log")

    # --- Lógica de valores por defecto dinámicos ---
    if cfg.youtube_url:
        logger.info("Modo YouTube detectado → Configurando defaults de YT")
        # Forzamos 0.5 para YouTube (o el valor que prefieras)
        cfg.peak_height = 0.5 
        
        if cfg.vertical and cfg.vertical_mode == "twitch":
            cfg.vertical_mode = "center"
    else:
        logger.info("Modo Twitch detectado → Configurando defaults de Twitch")
        # Si el usuario NO pasó un valor manual por consola, bajamos a 0.10
        # Comparamos contra el default original (0.5) del ArgumentParser
        if cfg.peak_height == 0.5: 
            cfg.peak_height = 0.10
            logger.info("Ajustando peak_height automático a 0.10 para mejor detección")

    if cfg.reset:
        summary = clean_workspace(cfg, logger, mode=cfg.reset)
        if cfg.reset_only:
            logger.info("Reset-only: %s", summary)
            print(f"[reset] {summary['files']} archivos y {summary['dirs']} carpetas eliminadas")
            return

    if cfg.youtube_url and cfg.vertical and cfg.vertical_mode == "twitch":
        logger.info("Modo YouTube detectado → cambiando vertical_mode a 'center'")
        cfg.vertical_mode = "center"

    if cfg.youtube_url and cfg.peak_height >= 0.5:
        logger.info("Modo YouTube detectado → ajustando peak_height a 0.25")
        cfg.peak_height = 0.25

    try:
        if cfg.youtube_url:
            video_path = download_youtube_video(cfg.youtube_url, cfg, logger)

            vod = {
                "id": Path(video_path).stem,
                "title": Path(video_path).stem,
                "duration": "",
            }

            chat_path = cfg.output_dir / f"{vod['id']}_chat.json"
            chat_ok = False
            logger.info("Modo YouTube → sin chat")
        else:
            token = get_access_token(cfg, logger)
            user_id = get_user_id(cfg, token, logger)

            vod = (
                {
                    "id": cfg.vod_id,
                    "url": f"https://www.twitch.tv/videos/{cfg.vod_id}",
                    "title": cfg.vod_id,
                    "created_at": "",
                }
                if cfg.vod_id
                else get_latest_vod(user_id, cfg, token, logger)
            )

            video_path = download_vod_if_needed(vod, cfg, logger)

            chat_path = cfg.output_dir / f"{vod['id']}_chat.json"
            chat_ok = download_chat(vod["id"], cfg, logger, chat_path)
            if not chat_ok:
                logger.warning("Seguimos el pipeline sin chat para este VOD.")

        motion = analyze_motion(video_path, logger)
        audio_wav = extract_audio(video_path, logger)
        audio_scores = analyze_audio_emotion(audio_wav, logger)
        segments = transcribe(video_path, cfg, logger)
        text_scores, chat_scores = score_text_and_chat(segments, chat_path, logger)

        clips = merge_signals_and_pick_clips(
            motion,
            audio_scores,
            text_scores,
            chat_scores,
            cfg,
            logger,
        )

        clips_dir = cfg.output_dir / str(vod["id"]) / "clips"
        for clip in clips:
            if cfg.preview_vertical and cfg.vertical:
                render_vertical_preview_frame(clip, video_path, clips_dir, cfg, logger)
            else:
                render_clip_with_subs(
                    clip,
                    video_path,
                    segments,
                    clips_dir,
                    cfg,
                    logger,
                )

        write_report(vod, clips, cfg, cfg.output_dir / str(vod["id"]), logger)

    except Exception as exc:
        logger.exception("Error durante la ejecución: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()