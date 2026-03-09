import argparse
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from utils import (Logger, print_title, calculate_time, get_image_paths,
                   image_process, image_process_image, download_from_modelscope,
                   load_model_config)

# SFW 系统提示词（纯净描述，用于反推训练）
DEFAULT_SYSTEM_PROMPT = """You are a professional image captioning assistant. Describe the image in detail, including subjects, actions, scene, objects, and any visible text. Use clear and concise natural language. Do not include any evaluations, opinions, or markdown formatting. The description should be suitable for training image generation models."""
DEFAULT_USER_PROMPT = "Please describe this image."


class Qwen3_5:
    def __init__(self, logger: Logger, model_path_or_id: str, args, is_local: bool = False):
        self.logger = logger
        self.model_path = model_path_or_id
        self.is_local = is_local
        self.args = args
        self.processor = None
        self.model = None

    def load_model(self):
        self.logger.info(f"Loading Qwen3.5 model from {self.model_path}...")
        start = time.monotonic()

        # 量化配置
        quant_config = None
        compute_dtype = torch.float16
        if self.args.llm_qnt == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )
        elif self.args.llm_qnt == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

        device = "cpu" if self.args.llm_use_cpu else "auto"
        attn_impl = "sdpa" if self.args.use_flash_attn else None

        # 加载模型
        load_kwargs = {
            "device_map": device,
            "quantization_config": quant_config,
            "trust_remote_code": True,
            "local_files_only": self.is_local,
            "attn_implementation": attn_impl,
        }
        if quant_config is not None:
            load_kwargs["torch_dtype"] = compute_dtype

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.is_local
        )
        self.logger.info(f"Model loaded in {time.monotonic()-start:.1f}s")

    def get_caption(self, image: Image.Image, system_prompt: str, user_prompt: str,
                    temperature: float, top_p: float, max_new_tokens: int) -> str:
        return self.get_caption_batch([image], system_prompt, user_prompt,
                                      temperature, top_p, max_new_tokens)[0]

    def get_caption_batch(self, images: List[Image.Image], system_prompt: str, user_prompt: str,
                          temperature: float, top_p: float, max_new_tokens: int) -> List[str]:
        if self.model is None:
            self.load_model()
        if not self.args.llm_use_cpu:
            torch.cuda.empty_cache()

        # 预处理所有图像
        processed_images = []
        for img in images:
            img_np = image_process(img, self.args.image_size)
            processed_images.append(image_process_image(img_np))

        # 构建消息模板（第一条用于获取文本）
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        })
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        # 批量处理
        inputs = self.processor(
            images=processed_images,
            text=[text] * len(processed_images),
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 生成参数（只传递大于0的值，表示使用模型默认）
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens > 0 else {}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        if top_p > 0:
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, **gen_kwargs)
            # 去除输入部分
            gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
            captions = self.processor.batch_decode(gen_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)

        # 根据开关去除思考标签
        if self.args.disable_think:
            captions = [re.sub(r'<think>.*?</think>\s*', '', cap, flags=re.DOTALL).strip() for cap in captions]
        return captions

    def unload_model(self):
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
        if not self.args.llm_use_cpu:
            torch.cuda.empty_cache()
        self.logger.info("Model unloaded")


class Caption:
    def __init__(self):
        self.logger = None
        self.model = None

    def check_path(self, args):
        if not args.data_path:
            raise ValueError("data_path required")
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"{args.data_path} not found")

    def set_logger(self, args):
        log_file = None
        if args.save_logs:
            data_dir = Path(args.data_path)
            base = data_dir.parent if data_dir.parent.exists() else Path.cwd()
            if args.custom_caption_save_path:
                base = Path(args.custom_caption_save_path)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = data_dir.name if data_dir.exists() else "caption"
            log_file = base / f"Caption_{log_name}_{ts}.log"
        self.logger = Logger(args.log_level, str(log_file) if log_file else None)
        if log_file:
            self.logger.info(f"Log saved to {log_file}")

    def load_models(self, args):
        if not args.llm_model_name.startswith("Qwen/"):
            config = load_model_config(Path(__file__).parent / "configs" / "default_qwen_vl.json")
            if args.llm_model_name in config:
                args.llm_model_name = config[args.llm_model_name]
            else:
                raise ValueError(f"Unknown model name: {args.llm_model_name}")

        if args.model_site == "modelscope":
            try:
                local_path = download_from_modelscope(args.llm_model_name, cache_dir=args.models_cache_dir)
                self.model = Qwen3_5(self.logger, local_path, args, is_local=True)
            except Exception as e:
                self.logger.error(f"ModelScope download failed: {e}, falling back to HuggingFace")
                self.model = Qwen3_5(self.logger, args.llm_model_name, args, is_local=False)
        else:
            self.model = Qwen3_5(self.logger, args.llm_model_name, args, is_local=False)
        self.model.load_model()

    def run_inference(self, args):
        start = time.monotonic()
        image_paths = get_image_paths(self.logger, Path(args.data_path), args.recursive)
        total = len(image_paths)
        if total == 0:
            self.logger.warning("No images found.")
            return

        batch_size = args.batch_size
        self.logger.info(f"Processing {total} images with batch size {batch_size}...")
        pbar = tqdm(total=total, smoothing=0.0)

        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            try:
                # 准备图像和对应的保存路径
                images = []
                cap_files = []
                for img_path in batch_paths:
                    # 确定 caption 文件路径
                    if args.custom_caption_save_path:
                        rel = os.path.splitext(os.path.relpath(img_path, args.data_path))[0]
                        rel = rel[1:] if rel.startswith('/') else rel
                        cap_dir = Path(args.custom_caption_save_path) / os.path.dirname(rel)
                        cap_dir.mkdir(parents=True, exist_ok=True)
                        cap_file = cap_dir / (os.path.basename(rel) + args.caption_extension)
                    else:
                        cap_file = Path(os.path.splitext(img_path)[0] + args.caption_extension)

                    if args.skip_exists and cap_file.exists():
                        self.logger.warning(f"Skip existing {cap_file}")
                        continue  # 跳过该图片

                    images.append(Image.open(img_path))
                    cap_files.append(cap_file)

                if not images:
                    pbar.update(len(batch_paths))
                    continue

                # 批量推理
                captions = self.model.get_caption_batch(
                    images=images,
                    system_prompt=args.llm_system_prompt,
                    user_prompt=args.llm_user_prompt,
                    temperature=args.llm_temperature,
                    top_p=args.llm_top_p,
                    max_new_tokens=args.llm_max_tokens
                )

                # 保存
                for cap_file, caption in zip(cap_files, captions):
                    if not (args.not_overwrite and cap_file.exists()):
                        with open(cap_file, "w", encoding="utf-8") as f:
                            f.write(caption + "\n")
                        self.logger.debug(f"Saved to {cap_file}")
                    else:
                        self.logger.warning(f"Not overwriting {cap_file}")

            except Exception as e:
                self.logger.error(f"Batch failed: {e}")
                # 仍然更新进度条
            finally:
                pbar.update(len(batch_paths))

        pbar.close()
        self.logger.info(f"Done in {calculate_time(start)}")

    def unload_models(self):
        if self.model:
            self.model.unload_model()


def setup_args():
    parser = argparse.ArgumentParser()
    base = parser.add_argument_group("Base")
    base.add_argument('--data_path', type=str, required=True, help='Path to image or directory')
    base.add_argument('--recursive', action='store_true', help='Search recursively')
    base.add_argument('--caption_extension', type=str, default='.txt', help='Caption file extension')
    base.add_argument('--custom_caption_save_path', type=str, help='Custom save directory')

    log = parser.add_argument_group("Logging")
    log.add_argument('--log_level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    log.add_argument('--save_logs', action='store_true', help='Save log file')

    model = parser.add_argument_group("Model")
    model.add_argument('--model_site', type=str, choices=['huggingface', 'modelscope'], default='modelscope')
    model.add_argument('--models_cache_dir', type=str, default=None, help='Cache directory')
    model.add_argument('--llm_model_name', type=str, required=True, help='Model ID or display name')
    model.add_argument('--llm_use_cpu', action='store_true')
    model.add_argument('--llm_qnt', type=str, choices=['none', '4bit', '8bit'], default='none')
    model.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention 2 if available')
    model.add_argument('--disable_think', action='store_true', help='Remove thinking tags from output')
    model.add_argument('--llm_system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT)
    model.add_argument('--llm_user_prompt', type=str, default=DEFAULT_USER_PROMPT)
    model.add_argument('--llm_temperature', type=float, default=0.0,
                       help='Temperature (0 = use model default)')
    model.add_argument('--llm_top_p', type=float, default=0.0,
                       help='Top-p (0 = use model default)')
    # ========== 修改点：默认值改为1024，说明保留0为使用模型默认 ==========
    model.add_argument('--llm_max_tokens', type=int, default=1024,
                       help='Max new tokens (0 = use model default)')
    model.add_argument('--image_size', type=int, default=1024, help='Resize image to this size')

    batch = parser.add_argument_group("Batch")
    batch.add_argument('--batch_size', type=int, default=1, help='Number of images to process in parallel')

    misc = parser.add_argument_group("Misc")
    misc.add_argument('--skip_exists', action='store_true')
    misc.add_argument('--not_overwrite', action='store_true')

    # GUI dummy args
    parser.add_argument('--theme', type=str, choices=['base', 'ocean', 'origin'], default='base')
    parser.add_argument('--port', type=int, default=8282)
    parser.add_argument('--listen', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--inbrowser', action='store_true')

    return parser.parse_args()


def main():
    print_title()
    args = setup_args()
    if not args.llm_model_name.startswith("Qwen/"):
        config = load_model_config(Path(__file__).parent / "configs" / "default_qwen_vl.json")
        if args.llm_model_name in config:
            args.llm_model_name = config[args.llm_model_name]
        else:
            raise ValueError(f"Unknown model name: {args.llm_model_name}")
    cap = Caption()
    cap.check_path(args)
    cap.set_logger(args)
    cap.load_models(args)
    cap.run_inference(args)
    cap.unload_models()


if __name__ == "__main__":
    main()