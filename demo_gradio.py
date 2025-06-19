import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import logging

DEFAULT_PIPELINE_PATH = "black-forest-labs/FLUX.1-dev"
DEFAULT_QWEN_MODEL_PATH = "Qwen/Qwen3-8B"
DEFAULT_CUSTOM_WEIGHTS_PATH = "PosterCraft/PosterCraft-v1_RL"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Qwen Recap Agent ---
class QwenRecapAgent:
    def __init__(self, model_path, max_retries=3, retry_delay=2, device_map="auto"):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_kwargs = {"torch_dtype": "auto", "device_map": device_map if device_map == "auto" else None}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device_map != "auto":
             self.model.to(device_map)
        
        self.prompt_template = """You are an expert poster prompt designer. Your task is to rewrite a user's short poster prompt into a detailed and vivid long-format prompt. Follow these steps carefully:

    **Step 1: Analyze the Core Requirements**
    Identify the key elements in the user's prompt. Do not miss any details.
    - **Subject:** What is the main subject? (e.g., a person, an object, a scene)
    - **Style:** What is the visual style? (e.g., photorealistic, cartoon, vintage, minimalist)
    - **Text:** Is there any text, like a title or slogan?
    - **Color Palette:** Are there specific colors mentioned?
    - **Composition:** Are there any layout instructions?

    **Step 2: Expand and Add Detail**
    Elaborate on each core requirement to create a rich description.
    - **Do Not Omit:** You must include every piece of information from the original prompt.
    - **Enrich with Specifics:** Add professional and descriptive details.
        - **Example:** If the user says "a woman with a bow", you could describe her as "a young woman with a determined expression, holding a finely crafted wooden longbow, with an arrow nocked and ready to fire."
    - **Fill in the Gaps:** If the original prompt is simple (e.g., "a poster for a coffee shop"), use your creativity to add fitting details. You might add "The poster features a top-down view of a steaming latte with delicate art on its foam, placed on a rustic wooden table next to a few scattered coffee beans."

    **Step 3: Handle Text Precisely**
    - **Identify All Text Elements:** Carefully look for any text mentioned in the prompt. This includes:
        - **Explicit Text:** Subtitles, slogans, or any text in quotes.
        - **Implicit Titles:** The name of an event, movie, or product is often the main title. For example, if the prompt is "generate a 'Inception' poster ...", the title is "Inception".
    - **Rules for Text:**
        - **If Text Exists:**
            - You must use the exact text identified from the prompt.
            - Do NOT add new text or delete existing text.
            - Describe each text's appearance (font, style, color, position). Example: `The title 'Inception' is written in a bold, sans-serif font, integrated into the cityscape.`
        - **If No Text Exists:**
            - Do not add any text elements. The poster must be purely visual.
    - Most posters have titles. When a title exists, you must extend the title's description. Only when you are absolutely sure that there is no text to render, you can allow the extended prompt not to render text.

    **Step 4: Final Output Rules**
    - **Output ONLY the rewritten prompt.** No introductions, no explanations, no "Here is the prompt:".
    - **Use a descriptive and confident tone.** Write as if you are describing a finished, beautiful poster.
    - **Keep it concise.** The final prompt should be under 300 words.

    ---
    **User Prompt:**
    {brief_description}"""
    
    def recap_prompt(self, original_prompt):
        full_prompt = self.prompt_template.format(brief_description=original_prompt)
        messages = [{"role": "user", "content": full_prompt}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096, temperature=0.6)
            
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            full_response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            final_answer = self._extract_final_answer(full_response)
            
            if final_answer:
                return final_answer.strip()
            
            print("Qwen returned an empty answer. Using original prompt.")
            return original_prompt
        except Exception as e:
            print(f"Qwen recap failed: {e}. Using original prompt.")
            return original_prompt

    def _extract_final_answer(self, full_response):
        if "</think>" in full_response:
            return full_response.split("</think>")[-1].strip()
        if "<think>" not in full_response:
            return full_response.strip()
        return None

# --- Global State and Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logging.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
else:
    logging.warning("CUDA not available. Falling back to CPU.")


# --- Poster Generator Class ---
class PosterGenerator:
    def __init__(self, pipeline_path, qwen_model_path, custom_weights_path, device):
        self.device = device
        self.qwen_agent = self._load_qwen_agent(qwen_model_path)
        self.pipeline = self._load_flux_pipeline(pipeline_path, custom_weights_path)
        
    def _load_qwen_agent(self, qwen_model_path):
        if not qwen_model_path:
            return None
        return QwenRecapAgent(model_path=qwen_model_path, device_map=str(self.device))

    def _load_flux_pipeline(self, pipeline_path, custom_weights_path):
        pipeline = FluxPipeline.from_pretrained(pipeline_path, torch_dtype=torch.bfloat16)
        
        if custom_weights_path and os.path.exists(custom_weights_path):
            logging.info(f"Loading custom Transformer from directory: {custom_weights_path}")
            transformer = FluxTransformer2DModel.from_pretrained(
                custom_weights_path, torch_dtype=torch.bfloat16
            )
            pipeline.transformer = transformer
        
        pipeline.to(self.device)
        return pipeline

    def generate(self, prompt, enable_recap, **kwargs):
        final_prompt = prompt
        if enable_recap:
            if not self.qwen_agent:
                raise gr.Error("Recap is enabled, but the recap model is not available. Check model path.")
            final_prompt = self.qwen_agent.recap_prompt(prompt)

        generator = torch.Generator(device=self.device).manual_seed(kwargs['seed'])
        
        with torch.inference_mode():
            image = self.pipeline(
                prompt=final_prompt,
                generator=generator,
                num_inference_steps=kwargs['num_inference_steps'],
                guidance_scale=kwargs['guidance_scale'],
                width=kwargs['width'],
                height=kwargs['height']
            ).images[0]
        
        return image, final_prompt

# --- Global Model Initialization ---
generator = PosterGenerator(
    pipeline_path=DEFAULT_PIPELINE_PATH,
    qwen_model_path=DEFAULT_QWEN_MODEL_PATH,
    custom_weights_path=DEFAULT_CUSTOM_WEIGHTS_PATH,
    device=device
)

# --- Gradio Interface Logic ---
def generate_image_interface(
    original_prompt, enable_recap, height, width, 
    num_inference_steps, guidance_scale, seed_input
):
    if not original_prompt or not original_prompt.strip():
        raise gr.Error("Prompt cannot be empty!")

    try:
        actual_seed = int(seed_input) if seed_input and seed_input > 0 else random.randint(1, 2**32 - 1)
        
        image, final_prompt = generator.generate(
            prompt=original_prompt,
            enable_recap=enable_recap,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=actual_seed
        )
        
        status_log = f"Seed: {actual_seed} | Generation complete."
        return image, final_prompt, status_log

    except Exception as e:
        raise gr.Error(f"An error occurred: {e}")


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PosterCraft-v1.0")
    gr.Markdown(f"Running on: **{device}** | Base Pipeline: **{DEFAULT_PIPELINE_PATH}**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your creative prompt...")
            enable_recap_checkbox = gr.Checkbox(label="Enable Prompt Recap", value=True, info=f"Uses {DEFAULT_QWEN_MODEL_PATH} for rewriting.")
            
            with gr.Row():
                width_input = gr.Slider(label="Width", minimum=256, maximum=2048, value=832, step=64)
                height_input = gr.Slider(label="Height", minimum=256, maximum=2048, value=1216, step=64)
            gr.Markdown("Tip: Recommended size is 832x1216 for best results.")
            
            num_inference_steps_input = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=28, step=1)
            guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=0.0, maximum=20.0, value=3.5, step=0.1)
            seed_number_input = gr.Number(label="Seed", value=None, minimum=-1, step=1, info="Leave blank or set to -1 for a random seed.")
            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. Results")
            image_output = gr.Image(label="Generated Image", type="pil", show_download_button=True, height=512)
            recapped_prompt_output = gr.Textbox(label="Final Prompt Used", lines=5, interactive=False)
            status_output = gr.Textbox(label="Status Log", lines=4, interactive=False)

    inputs_list = [
        prompt_input, enable_recap_checkbox, height_input, width_input,
        num_inference_steps_input, guidance_scale_input, seed_number_input
    ]
    outputs_list = [image_output, recapped_prompt_output, status_output]
    
    generate_button.click(fn=generate_image_interface, inputs=inputs_list, outputs=outputs_list)

if __name__ == "__main__":
    logging.info("Starting Gradio application...")
    demo.queue().launch(server_name="0.0.0.0", server_port=8420, share=False)
