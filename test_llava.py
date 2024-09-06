"""Modified from llava.eval.run_llava.py to test inference w/ smaller GPUs (e.g. 16GB Titan V)"""

import argparse
import torch
from typing import Optional

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(
    model_path: str,
    image_files: list[str],
    query: str,
    model_base: Optional[str] = None,
    conv_mode: Optional[str] = None,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    load_4bit: bool = False,
    load_8bit: bool = False,
    device: str = "cuda",
    use_flash_attn: bool = True,
):
    assert not (load_4bit and load_8bit), "Cannot use both load_4bit=True and load_8bit=True"

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device,
        use_flash_attn=use_flash_attn,
    )

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image-files",
        type=str,
        nargs="+",
        default=("https://llava-vl.github.io/static/images/view.jpg",),
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the things I should be cautious about when I visit here?",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,  # Quickstart guide used 0
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # For lora need `model_base`, for others it is None
    # Value from comment below table in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15
    if "lora" in args.model_path and args.model_base is None:
        if args.model_path == "liuhaotian/llava-v1.5-7b-lora":
            model_base = "lmsys/vicuna-7b-v1.5"
        elif args.model_path == "liuhaotian/llava-v1.5-13b-lora":
            model_base = "lmsys/vicuna-13b-v1.5"
        else:
            raise RuntimeError(
                "Unrecognized lora model path for automatically setting model_base."
                " Pass in --model_base manually."
            )

    eval_model(
        model_path=args.model_path,
        image_files=args.image_files,
        query=args.query,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        use_flash_attn=args.use_flash_attn,
    )
