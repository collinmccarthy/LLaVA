"""Modified from llava.eval.run_llava.py to test inference w/ smaller GPUs (e.g. 16GB Titan V)"""

import argparse
import torch
import warnings
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
    download_only: bool = False,
    batch_inference: bool = False,
    print_inputs: bool = False,
    extra_image_grid_pinpoints: Optional[list[tuple[int, int]]] = None,
) -> None:
    assert not (load_4bit and load_8bit), "Cannot use both load_4bit=True and load_8bit=True"

    # For lora need `model_base`, for others it is None
    # Value from comment below table in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15
    if "lora" in model_path and model_base is None:
        if model_path == "liuhaotian/llava-v1.5-7b-lora":
            model_base = "lmsys/vicuna-7b-v1.5"
        elif model_path == "liuhaotian/llava-v1.5-13b-lora":
            model_base = "lmsys/vicuna-13b-v1.5"
        else:
            raise RuntimeError(
                "Unrecognized lora model path for automatically setting model_base."
                " Pass in --model_base manually."
            )

        if load_4bit or load_8bit:
            warnings.warn(
                "Disabling load_4bit and load_8bit for loading LoRa models. See"
                " https://github.com/haotian-liu/LLaVA/issues/744#issuecomment-1793596446 for"
                " how to convert the checkpoint for quantized inference with LoRa."
            )
            load_4bit = False
            load_8bit = False

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

    if download_only:
        return

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

    if extra_image_grid_pinpoints is not None:
        if hasattr(model.config, "image_grid_pinpoints"):
            model.config.image_grid_pinpoints += extra_image_grid_pinpoints
        else:
            print(
                f"[WARNING] model.config does not have image_grid_pointpoints, ignoring"
                f" extra_image_grid_pinpoints={extra_image_grid_pinpoints}."
            )

    images_tensor = process_images(images, image_processor, model.config)
    assert isinstance(images_tensor, torch.Tensor)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    assert isinstance(input_ids, torch.Tensor)
    input_ids = input_ids.unsqueeze(0).to(model.device)

    # Added following https://github.com/haotian-liu/LLaVA/pull/1502/files#diff-a811990636acaacacca7288cd7ba5b220e4b8b10aff80f904ebca0349c6e02e4
    if batch_inference:
        input_ids = torch.cat([input_ids for _ in images]).to(model.device)

    # Original generate method
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
            # DEBUG ONLY
            tokenizer=tokenizer if print_inputs else None,
        )

    # Added following https://github.com/haotian-liu/LLaVA/pull/1502/files#diff-a811990636acaacacca7288cd7ba5b220e4b8b10aff80f904ebca0349c6e02e4
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for i, image_file in enumerate(image_files):
        print("-" * 80)
        print(f"Image [{i+1}/{len(image_files)}]: {image_file}")
        print("- " * 40)
        print(outputs[i].strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        "--model_path",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
    )
    parser.add_argument(
        "--model-base",
        "--model_base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image-files",
        "--image_files",
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
        "--conv_mode",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,  # Quickstart guide used 0
    )
    parser.add_argument(
        "--top-p",
        "--top_p",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num-beams",
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--load-4bit",
        "--load_4bit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load-8bit",
        "--load_8bit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-flash-attn",
        "--use_flash_attn",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--download-model-paths",
        "--download_model_paths",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-batch",
        "--no_batch",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--print-inputs",
        "--print_inputs",
        action="store_true",
        default=False,
        help="Print inputs during processing to aid in debugging.",
    )
    parser.add_argument(
        "--extend-anyres-shape",
        "--extend_anyres_shape",
        type=int,
        nargs="+",
        help="H and W values to override model.config.image_grid_pinpoints",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.download_model_paths is not None:
        for model_path in args.download_model_paths:
            eval_model(
                model_path=model_path,
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
                device="cpu",  # Download only, model doesn't need to be on GPU
                download_only=True,
            )

        print(
            f"Finished downloading models, exiting. To run a given model use --model-path instead"
            f" of --download-model-paths"
        )

    else:
        if len(args.image_files) > 1:
            if args.no_batch is None and args.batch is None:
                # Default is to use batch inference if multiple image files
                print(
                    "Using batch=True for batch inference with multiple image files. To process"
                    " images in a single prompt pass in --no-batch"
                )
                batch_inference = True
            elif args.no_batch and not args.batch:
                batch_inference = False
            elif args.batch and not args.no_batch:
                batch_inference = True
            else:
                assert False, "Cannot have both --batch and --no-batch"
        else:
            batch_inference = False

        extra_image_grid_pinpoints = None
        if args.extend_anyres_shape is not None:
            assert len(args.extend_anyres_shape) == 2, "Expected 2-tuple for --extend-anyres-shape"
            extra_image_grid_pinpoints = [args.extend_anyres_shape]

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
            batch_inference=batch_inference,
            print_inputs=args.print_inputs,
            extra_image_grid_pinpoints=extra_image_grid_pinpoints,
        )
