import os
os.environ['HF_HOME'] = '/nfs/stak/users/lowecal/hpc-share/huggingface'

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch

class molmo_handler:
    def __init__(self, model_name: str = 'allenai/Molmo-7B-D-0924', cache_dir: str = "/nfs/stak/users/lowecal/hpc-share/vlm-testing/model_files"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
    
    def generate_text(self, image_path: str, prompt: str) -> str:
        # Process the image and text
        inputs = self.processor.process(
            images=[Image.open(image_path)],
            text=prompt
        )

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
            tokenizer=self.processor.tokenizer
        )

        # Only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text
