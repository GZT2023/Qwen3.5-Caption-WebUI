import argparse
import os
import time
from pathlib import Path

import gradio as gr
from PIL import Image

import caption
from utils import print_title, load_model_config

CONFIG_FILE = Path(__file__).parent / "configs" / "default_qwen_vl.json"
IS_MODEL_LOAD = False
ARGS = None
CAPTION_FN = None


def gui_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str, default="base", choices=["base", "ocean", "origin"])
    parser.add_argument('--port', type=int, default=8282)
    parser.add_argument('--listen', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--inbrowser', action='store_true')
    parser.add_argument('--log_level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    parser.add_argument('--models_save_path', type=str, default=os.path.join(os.getcwd(), "models"),
                        help='Directory to save models')
    return parser.parse_args()


def gui():
    global IS_MODEL_LOAD, ARGS, CAPTION_FN
    print_title()

    get_gui_args = gui_setup_args()
    if get_gui_args.theme == "ocean":
        theme = gr.themes.Ocean()
    elif get_gui_args.theme == "origin":
        theme = gr.themes.Origin()
    else:
        theme = gr.themes.Base()

    model_config = load_model_config(CONFIG_FILE)

    with gr.Blocks(title="Qwen3.5 Caption WebUI") as demo:
        gr.Markdown("## Qwen3.5 Caption WebUI")
        with gr.Row():
            close_btn = gr.Button("Close Server", variant="primary")

        with gr.Row():
            with gr.Column():
                # 模型选择
                model_choice = gr.Dropdown(
                    label="Qwen3.5 Model",
                    choices=list(model_config.keys()),
                    value=list(model_config.keys())[0]
                )

                # 下载站点
                site_choice = gr.Radio(
                    label="Download Source",
                    choices=["modelscope", "huggingface"],
                    value="modelscope"
                )

                with gr.Row():
                    use_cpu = gr.Checkbox(label="Use CPU")
                    quant = gr.Radio(label="Quantization", choices=["none", "4bit", "8bit"], value="none")

                # 新增选项
                with gr.Row():
                    use_flash = gr.Checkbox(label="Use SDPA (PyTorch native attention, more stable)", value=False)
                    disable_think = gr.Checkbox(label="Disable thinking mode", value=True)

                load_btn = gr.Button("Load Model", variant="primary")
                unload_btn = gr.Button("Unload Model")

                with gr.Accordion("Generation Parameters", open=False):
                    temp = gr.Slider(label="Temperature (0=model default)", minimum=0.0, maximum=1.0, value=0.0, step=0.1)
                    top_p = gr.Slider(label="Top_p (0=model default)", minimum=0.0, maximum=1.0, value=0.0, step=0.1)
                    # ========== 修改点：最大值改为8192，默认值1024 ==========
                    max_tokens = gr.Slider(label="Max New Tokens (0=use model default)", minimum=0, maximum=8192, value=1024, step=1)
                    img_size = gr.Slider(label="Image Size", minimum=256, maximum=2048, value=1024, step=1)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=64, value=1, step=1, info="Number of images to process in parallel")
                    auto_unload = gr.Checkbox(label="Auto unload after inference")

                sys_prompt = gr.Textbox(label="System Prompt", lines=3, value=caption.DEFAULT_SYSTEM_PROMPT)
                user_prompt = gr.Textbox(label="User Prompt", lines=2, value=caption.DEFAULT_USER_PROMPT)

            with gr.Column():
                with gr.Tab("Single Image"):
                    img_input = gr.Image(type="filepath", label="Upload Image")
                    submit_btn = gr.Button("Generate Caption", variant="primary")
                    output = gr.Textbox(label="Caption", lines=10, show_copy_button=True)

                with gr.Tab("Batch"):
                    input_dir = gr.Textbox(label="Input Directory")
                    recursive = gr.Checkbox(label="Recursive")
                    custom_save = gr.Textbox(label="Custom Save Directory")
                    with gr.Row():
                        skip = gr.Checkbox(label="Skip if exists")
                        no_overwrite = gr.Checkbox(label="Do not overwrite")
                    ext = gr.Textbox(label="Caption Extension", value=".txt")
                    batch_btn = gr.Button("Start Batch", variant="primary")

        # 加载模型
        def load_models(model_display_name, site, cpu, quant_val, use_flash, disable_think_val):
            global IS_MODEL_LOAD, ARGS, CAPTION_FN
            if IS_MODEL_LOAD:
                gr.Warning("Model already loaded!")
                return [gr.update()]*17

            real_model_id = model_config[model_display_name]

            if ARGS is None:
                gui_args = gui_setup_args()
                ARGS = argparse.Namespace()
                ARGS.save_logs = False
                ARGS.log_level = gui_args.log_level
                ARGS.custom_caption_save_path = None
                ARGS.caption_extension = '.txt'
                ARGS.skip_exists = False
                ARGS.not_overwrite = False
                ARGS.recursive = False
                ARGS.data_path = None
                ARGS.models_cache_dir = gui_args.models_save_path
                ARGS.llm_system_prompt = caption.DEFAULT_SYSTEM_PROMPT
                ARGS.llm_user_prompt = caption.DEFAULT_USER_PROMPT
                ARGS.llm_temperature = 0.0
                ARGS.llm_top_p = 0.0
                # ========== 修改点：与滑块默认值保持一致 ==========
                ARGS.llm_max_tokens = 1024
                ARGS.image_size = 1024
                ARGS.batch_size = 1
                ARGS.llm_use_cpu = False
                ARGS.llm_qnt = 'none'
                ARGS.use_flash_attn = False
                ARGS.disable_think = True
                ARGS.model_site = site

            args = ARGS
            args.model_site = site
            args.llm_model_name = real_model_id
            args.llm_use_cpu = cpu
            args.llm_qnt = quant_val
            args.use_flash_attn = use_flash
            args.disable_think = disable_think_val

            if CAPTION_FN is None:
                CAPTION_FN = caption.Caption()
                CAPTION_FN.set_logger(args)
            CAPTION_FN.load_models(args)
            IS_MODEL_LOAD = True
            gr.Info("Model loaded")

            # 定义需要禁用的控件索引（从0开始）
            # model_choice (0), site_choice (1), use_cpu (2), quant (3), use_flash (4), disable_think (5)
            # 这些是加载选项，应禁用
            # temp (6), top_p (7), max_tokens (8), img_size (9), batch_size (10), auto_unload (11),
            # sys_prompt (12), user_prompt (13) 这些是生成参数，保持可交互
            # 因此前6个禁用，第6-13保持交互，后两个按钮 load_btn (14), unload_btn (15), output (16)
            updates = []
            # 前6个禁用
            for i in range(6):
                updates.append(gr.update(interactive=False))
            # 中间8个保持交互（不更新 interactive 或设为 True）
            for i in range(6, 14):
                updates.append(gr.update(interactive=True))
            # load_btn 变为 secondary
            updates.append(gr.update(variant="secondary"))
            # unload_btn 变为 primary
            updates.append(gr.update(variant="primary"))
            # output 保持不变
            updates.append(gr.update())
            return updates

        load_outputs = [model_choice, site_choice, use_cpu, quant, use_flash, disable_think,
                        temp, top_p, max_tokens, img_size, batch_size, auto_unload,
                        sys_prompt, user_prompt, load_btn, unload_btn, output]

        load_btn.click(fn=load_models,
                       inputs=[model_choice, site_choice, use_cpu, quant, use_flash, disable_think],
                       outputs=load_outputs)

        # 卸载模型
        def unload_models():
            global IS_MODEL_LOAD
            if IS_MODEL_LOAD:
                CAPTION_FN.unload_models()
                IS_MODEL_LOAD = False
                gr.Info("Model unloaded")
                # 恢复所有控件可交互
                updates = []
                # 前6个恢复交互
                for i in range(6):
                    updates.append(gr.update(interactive=True))
                # 中间8个保持交互（也可不更新，但为了统一设为True）
                for i in range(6, 14):
                    updates.append(gr.update(interactive=True))
                # load_btn 恢复 primary
                updates.append(gr.update(variant="primary"))
                # unload_btn 恢复 secondary
                updates.append(gr.update(variant="secondary"))
                # output 保持不变
                updates.append(gr.update())
                return updates
            else:
                gr.Warning("No model loaded")
                return [gr.update()] * 17

        unload_btn.click(fn=unload_models, outputs=load_outputs)

        # 单图推理
        def infer_single(img, model_display_name, site, cpu, quant_val, use_flash, disable_think_val,
                         temperature, top_p_val, max_tok, img_sz, batch_sz, auto_unload_flag,
                         sys_p, user_p):
            if not IS_MODEL_LOAD:
                raise gr.Error("Model not loaded")
            real_model_id = model_config[model_display_name]

            args = ARGS
            args.model_site = site
            args.llm_model_name = real_model_id
            args.llm_use_cpu = cpu
            args.llm_qnt = quant_val
            args.use_flash_attn = use_flash
            args.disable_think = disable_think_val
            args.llm_temperature = temperature
            args.llm_top_p = top_p_val
            args.llm_max_tokens = max_tok
            args.image_size = img_sz
            args.batch_size = batch_sz
            args.llm_system_prompt = sys_p
            args.llm_user_prompt = user_p

            image = Image.open(img)
            start = time.monotonic()
            cap = CAPTION_FN.model.get_caption(
                image, sys_p, user_p,
                temperature, top_p_val, max_tok
            )
            if auto_unload_flag:
                unload_models()
            return cap

        submit_btn.click(fn=infer_single,
                         inputs=[img_input, model_choice, site_choice, use_cpu, quant, use_flash, disable_think,
                                 temp, top_p, max_tokens, img_size, batch_size, auto_unload,
                                 sys_prompt, user_prompt],
                         outputs=output)

        # 批量处理
        def batch_process(btn_val, dir_path, rec, custom, skip_exist, no_over, ext_name,
                          model_display_name, site, cpu, quant_val, use_flash, disable_think_val,
                          temperature, top_p_val, max_tok, img_sz, batch_sz, auto_unload_flag,
                          sys_p, user_p):
            if btn_val != "Start Batch":
                return "Start Batch"
            if not dir_path:
                raise gr.Error("Input directory required")
            if not IS_MODEL_LOAD:
                raise gr.Error("Model not loaded")

            real_model_id = model_config[model_display_name]

            args = ARGS
            args.model_site = site
            args.llm_model_name = real_model_id
            args.data_path = dir_path
            args.recursive = rec
            args.custom_caption_save_path = custom
            args.skip_exists = skip_exist
            args.not_overwrite = no_over
            args.caption_extension = ext_name
            args.llm_use_cpu = cpu
            args.llm_qnt = quant_val
            args.use_flash_attn = use_flash
            args.disable_think = disable_think_val
            args.llm_temperature = temperature
            args.llm_top_p = top_p_val
            args.llm_max_tokens = max_tok
            args.image_size = img_sz
            args.batch_size = batch_sz
            args.llm_system_prompt = sys_p
            args.llm_user_prompt = user_p

            CAPTION_FN.run_inference(args)
            if auto_unload_flag:
                unload_models()
            return gr.update(value="Done!", variant="stop")

        batch_btn.click(fn=batch_process,
                        inputs=[batch_btn, input_dir, recursive, custom_save,
                                skip, no_overwrite, ext,
                                model_choice, site_choice, use_cpu, quant, use_flash, disable_think,
                                temp, top_p, max_tokens, img_size, batch_size, auto_unload,
                                sys_prompt, user_prompt],
                        outputs=batch_btn)

        def close_server():
            demo.close()
        close_btn.click(fn=close_server)

    demo.launch(
        server_name="0.0.0.0" if get_gui_args.listen else None,
        server_port=get_gui_args.port,
        share=get_gui_args.share,
        inbrowser=get_gui_args.inbrowser
    )


if __name__ == "__main__":
    gui()