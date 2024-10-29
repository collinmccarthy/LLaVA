# Open-LLaVA-NeXt: Code Changes

Manual diff with LLaVA repo by merging these files into main and removing whitespace-only and import-order-only changes.

- `conversation.py` (orig Open-LLaVA-NeXt change)
    - Added `conv_llava_llama_3` conversation (use "llava_llama_3" for --conv-mode or --version args)
- `eva/model_vqa_loader.py` (orig Open-LLaVA-NeXt change)
    - Added `args.square_eval`
    - Added `pad_token_id=tokenizer.pad_token_id` to model.generate()
- `eval/model_vqa_mmbench.py` (orig Open-LLaVA-NeXt change)
    - Added `args.square_eval`
    - Added `pad_token_id=tokenizer.pad_token_id` to model.generate()
- `eval/model_vqa_science.py` (orig Open-LLaVA-NeXt change)
    - Added `args.square_eval`
    - Added `pad_token_id=tokenizer.pad_token_id` to model.generate()
- `eval/model_vqa.py` (orig Open-LLaVA-NeXt change)
    - Added `args.square_eval`
    - Added `pad_token_id=tokenizer.pad_token_id` to model.generate()
- `eval/run_llava.py` (orig Open-LLaVA-NeXt change)
    - Added `pad_token_id=tokenizer.pad_token_id` to model.generate()
- `model/llava_arch.py` (our change)
    - Returns attention mask if padded (necessary for v1.6 batch inference)
- `model/multimodal_encoder/clip_encoder.py`
    - Remove `@torch.no_grad()` from `CLIPVisionTower.forward()` to allow fine-tuning
- `train/llava_trainer.py`
    - Update `create_optimizer()` to support fine-tuning vision tower via `mm_vision_tower_lr` arg
    - Add `get_vision_tower_state_maybe_zero_3()`
        - Not currently used, but see `train/train.py` and `train/llava_trainer.py` for where `get_mm_adapter_state_maybe_zero_3` is used (this would be similar / the same)
- `train/train.py`
    - Add to `TrainingArguments`
        - `unfreeze_mm_vision_tower: bool = field(default=False)`
        - `lora_qv_proj_only: bool = False`
        - `mm_vision_tower_lr: Optional[float] = None`
    - Add `get_vision_tower_state_maybe_zero_3()` (following `get_mm_adapter_state_maybe_zero_3()`)
    - Add `qv_proj_only` param to `find_all_linear_names()`
        - Pass in `lora_qv_proj_only` to get target modules for LoraConfig
        - Default for training arg is False, not used anywhere currently
    - Add `preprocess_llama3()`
        - Uses `SeparatorStype.LLAMA_2` instead of `SeparatorStype.MPT`
        - Splits by `conv.sep2` instead of `conv.sep`
        - Other changes to how the targets are ignored
        - Calls `preprocess_llama3()` from `preprocess()` (if default conv version == "llama3")
        - Adds support for `image_aspect_ratio=anyres` in `LazySupervisedDataset` (fine-tuning)
            - Also stores 'image_size' field in data_dict
        - Adds `image_sizes` to output dict in `DataCollatorForSupervisedDataset` (fine-tuning)
        - Adds support for `unfreeze_mm_vision_tower` (calls `unfreeze_vit()`)
        - Prints total and trainable params
        - Adds `mm_vision_tower_lr` and `pad_token_id` to `model.config`
- `scripts/convert_seed_for_submission.py`
    - Remove `video` eval and remove outputting `result_upload_file` (not sure why)
- `scripts/v1_5`
    - Add `train/finetune.sh` file with srun command for fine-tuning v1.5
    - Change `--model_name_or_path liuhaotian/*` to `--model_name_or_path checkpoints/*`
        - `finetune_task.sh`, `finetune_task_lora.sh`
- `scripts/v1_5/eval`
    - Change `--model-path liuhaotian/*` to `--model-path checkpoints/*`
        - All `.sh` files?
- `scripts/v1_6`
    - Add `eval` dir with srun-style eval scripts like `scripts/v1_5/eval`
    - Add `train` dir with Vicuna-7B/LLaMA3-8B srun-style train scripts like `scripts/v1_5/train`

### Whitespace-Only Changes

- `eval/model_qa.py` (also changed order of imports)