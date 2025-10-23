import time
from argparse import Namespace
from pathlib import Path

from PIL import Image

from . import llm_inference, get_llm_dtype
from ..utils.image_process_util import image_process, image_process_image
from ..utils.logger_util import Logger


class QwenVL:
    def __init__(
            self,
            logger: Logger,
            models_paths: tuple[Path],
            args: Namespace,
    ):
        self.logger = logger
        self.args = args

        if len(models_paths) != 1:
            self.logger.error(self.logger.error(f"Invalid models paths: {models_paths}!!!"))
            raise ValueError

        self.llm_path = models_paths[0]
        self.llm_processor = None
        self.llm = None

    def load_model(self):
        # Import torch
        try:
            import torch
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError
        # Import transformers
        try:
            from transformers import (AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
                                      Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration,
                                      Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration)
        except ImportError as ie:
            self.logger.error(f'Import transformers Failed!\nDetails: {ie}')
            raise ImportError

        device = "cpu" if self.args.llm_use_cpu else "cuda"
        # Load LLM
        self.logger.info(
            f'Loading LLM `{self.args.llm_model_name}` with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        start_time = time.monotonic()
        # LLM dType
        llm_dtype = get_llm_dtype(logger=self.logger, args=self.args)
        self.logger.info(f'LLM dtype: {llm_dtype}')
        # LLM BNB quantization config
        if self.args.llm_qnt == "4bit":
            qnt_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=llm_dtype,
                                            bnb_4bit_use_double_quant=True)
            self.logger.info(f'LLM 4bit quantization: Enabled')
        elif self.args.llm_qnt == "8bit":
            qnt_config = BitsAndBytesConfig(load_in_8bit=True,
                                            llm_int8_enable_fp32_cpu_offload=True)
            self.logger.info(f'LLM 8bit quantization: Enabled')
        else:
            qnt_config = None
        # Load Qwen VL model
        if str(self.args.llm_model_name).startswith("Qwen3-VL"):
            if "A3B" in str(self.args.llm_model_name):
                self.llm = Qwen3VLMoeForConditionalGeneration.from_pretrained(self.llm_path,
                                                                              device_map="auto" \
                                                                                  if not self.args.llm_use_cpu else "cpu",
                                                                              torch_dtype=llm_dtype,
                                                                              quantization_config=qnt_config)
            else:
                self.llm = Qwen3VLForConditionalGeneration.from_pretrained(self.llm_path,
                                                                             device_map="auto" \
                                                                                 if not self.args.llm_use_cpu else "cpu",
                                                                             torch_dtype=llm_dtype,
                                                                             quantization_config=qnt_config)
        elif str(self.args.llm_model_name).startswith("Qwen2.5-VL"):
            self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.llm_path,
                                                                          device_map="auto" \
                                                                              if not self.args.llm_use_cpu else "cpu",
                                                                          torch_dtype=llm_dtype,
                                                                          quantization_config=qnt_config)
        else:
            self.llm = Qwen2VLForConditionalGeneration.from_pretrained(self.llm_path,
                                                                       device_map="auto" \
                                                                           if not self.args.llm_use_cpu else "cpu",
                                                                       torch_dtype=llm_dtype,
                                                                       quantization_config=qnt_config)
        self.llm.eval()
        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')
        # Load processor
        start_time = time.monotonic()
        self.logger.info(f'Loading processor with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        self.llm_processor = AutoProcessor.from_pretrained(self.llm_path)
        self.logger.info(f'Processor Loaded in {time.monotonic() - start_time:.1f}s.')

    def get_caption(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0,
            top_p: float = 0,
            max_new_tokens: int = 0,
    ) -> str:
        # Import torch
        try:
            import torch
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError
        with torch.no_grad():
            # Cleaning VRAM cache
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()

            image = image_process(image, target_size=int(self.args.image_size))
            self.logger.debug(f"Resized image shape: {image.shape}")
            image = image_process_image(image)

            if system_prompt:
                messages = [
                    {'role': 'system', 'content': f'{system_prompt}'},
                    {'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': f'{user_prompt}'}]
                     }
                ]
            else:
                self.logger.warning("System prompt NOT FOUND! Processing with out it.")
                messages = [
                    {'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': f'{user_prompt}'}]
                     }
                ]
            self.logger.debug(f"\nChat_template:\n{messages}")
            input_text = self.llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.llm_processor(image, input_text,
                                        # add_special_tokens=False,
                                        padding=True,
                                        return_tensors="pt").to(self.llm.device)
            # Generate caption
            if temperature == 0 and max_new_tokens == 0:
                max_new_tokens = 128
                self.logger.warning(f'LLM temperature and max_new_tokens not set, only '
                                    f'using default max_new_tokens value {max_new_tokens}')
                params = {
                    # 'greedy': 'false',
                    'top_p': 0.8,
                    'top_k': 20,
                    'temperature': 0.7,
                    # 'repetition_penalty': 1.0,
                    # 'presence_penalty': 1.5,
                    # 'out_seq_length': 16384
                }
            else:
                if temperature == 0:
                    temperature = 0.7
                    self.logger.warning(f'LLM temperature not set, using default value {temperature}')
                else:
                    self.logger.debug(f'LLM temperature is {temperature}')
                if top_p == 0:
                    top_p = 0.8
                    self.logger.warning(f'LLM top_p not set, using default value {top_p}')
                else:
                    self.logger.debug(f'LLM top_p is {top_p}')
                if max_new_tokens == 0:
                    max_new_tokens = 128
                    self.logger.warning(f'LLM max_new_tokens not set, using default value {max_new_tokens}')
                else:
                    self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
                params = {
                    # 'greedy': 'false',
                    'top_p': top_p,
                    'top_k': 20,
                    'temperature': temperature,
                    # 'repetition_penalty': 1.0,
                    # 'presence_penalty': 1.5,
                    # 'out_seq_length': 16384
                    # 'do_sample': True
                }

            generated_ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, **params)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.llm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            # content = self.llm_processor.decode(output[0][inputs["input_ids"].shape[-1]:],
            #                                     skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #
            # content_list = str(content).split(".")
            # unique_content = list(dict.fromkeys(content_list))
            # unique_content = '.'.join(unique_content)
            return output_text

    def inference(self):
        llm_inference(self)

    def unload_model(self) -> bool:
        llm_unloaded = False
        # Unload LLM
        if hasattr(self, "llm") and self.llm is not None:
            self.logger.info(f'Unloading LLM...')
            start = time.monotonic()
            del self.llm
            if hasattr(self, "llm_processer"):
                del self.llm_processor
            self.logger.info(f'LLM unloaded in {time.monotonic() - start:.1f}s.')
            llm_unloaded = True

        try:
            import torch
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError

        return llm_unloaded
