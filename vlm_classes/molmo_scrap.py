import os
os.environ['HF_HOME'] = '/nfs/stak/users/lowecal/hpc-share/huggingface'

import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch


def main(image_path: str, prompt: str):
    # Load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        cache_dir="model_files",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        cache_dir="/nfs/stak/users/lowecal/hpc-share/vlm-testing/model_files",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Process the image and textß
    inputs = processor.process(
        images=[Image.open(image_path)],
        text=prompt
    )

    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # Only get generated tokens; decode them to text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print the generated text
    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molmo-7B-D Image Captioning Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to describe the image")

    args = parser.parse_args()
    main(args.image_path, args.prompt)
