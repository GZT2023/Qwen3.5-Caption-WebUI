import os
import time
import base64
import glob
import logging
import json
from logging import handlers
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- 日志 ----------
class Logger:
    def __init__(self, level="INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if log_file:
            file_handler = handlers.TimedRotatingFileHandler(filename=log_file, when='D', interval=1,
                                                              backupCount=5, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            self.logger.warning("Log file not specified, only console output.")

    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)


def print_title():
    print("\n" + "*" * 70)
    print("********************* Qwen3.5 Caption WebUI *********************")
    print("*" * 70 + "\n")


def calculate_time(start_time: float) -> str:
    total = time.monotonic() - start_time
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days: parts.append(f"{days:.0f} Day(s)")
    if hours: parts.append(f"{hours:.0f} Hour(s)")
    if minutes: parts.append(f"{minutes:.0f} Min(s)")
    parts.append(f"{seconds:.2f} Sec(s)")
    return " ".join(parts)


# ---------- 图像处理 ----------
SUPPORT_IMAGE_FORMATS = ("bmp", "jpg", "jpeg", "png", "webp")

def get_image_paths(logger: Logger, path: Path, recursive: bool = False) -> List[str]:
    if os.path.isfile(path):
        if str(path).lower().endswith(SUPPORT_IMAGE_FORMATS):
            return [str(path)]
        else:
            logger.error("File is not an image.")
            raise FileNotFoundError
    pattern = os.path.join(path, '**') if recursive else os.path.join(path, '*')
    images = sorted(set(
        [img for img in glob.glob(pattern, recursive=recursive)
         if img.lower().endswith(SUPPORT_IMAGE_FORMATS)]
    ))
    logger.info(f'Found {len(images)} image(s).')
    return images

def image_process(image: Image.Image, target_size: int) -> np.ndarray:
    # RGBA -> white background
    image = image.convert('RGBA')
    new = Image.new('RGBA', image.size, 'WHITE')
    new.alpha_composite(image)
    image = new.convert('RGB')
    # Pad to square
    w, h = image.size
    desired = max(max(w, h), target_size)
    dw, dh = desired - w, desired - h
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img_arr = np.asarray(image)
    padded = cv2.copyMakeBorder(img_arr, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
    # Resize
    if padded.shape[0] > target_size:
        padded = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    elif padded.shape[0] < target_size:
        padded = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return padded

def image_process_image(padded: np.ndarray) -> Image.Image:
    return Image.fromarray(padded)

def encode_image_to_base64(image: Image.Image, fmt="PNG") -> str:
    with BytesIO() as buf:
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------- ModelScope 下载 ----------
def download_from_modelscope(model_id: str, cache_dir: Optional[str] = None) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        raise ImportError("请安装 modelscope: pip install modelscope")

    print(f"正在从 ModelScope 下载 {model_id} ...")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "modelscope")
        os.makedirs(cache_dir, exist_ok=True)
    local_path = snapshot_download(model_id, cache_dir=cache_dir, revision='master')
    print(f"模型下载完成: {local_path}")
    return local_path


def load_model_config(config_file: Path) -> dict:
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)