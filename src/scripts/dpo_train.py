import os
import sys
import json
from dataclasses import asdict
import importlib
import logging

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from trl import DPOTrainer
from utils.data_args import ScriptArguments
from utils.training_utils import (
    get_base_model,
    get_data_collator,
    upload_model_to_s3,
    save_model_wrapper,
    get_lora_and_quantization_configs,
    get_default_dtype,
)
import deepspeed

# Initiate logging
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables for distributed training set by torchrun
LOCAL_RANK = int(os.environ["LOCAL_RANK"])  # 0-7
RANK = int(os.environ["RANK"])  # 0-16, global rank
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])  # 8
WORLD_SIZE = int(os.environ["WORLD_SIZE"])  # 8x2, global world size
NODE_SIZE = WORLD_SIZE // LOCAL_WORLD_SIZE

logger.info(f"NCCL version is {torch.cuda.nccl.version()}")
logger.info(f"Torch supports architectures: {torch.cuda.get_arch_list()}")
logger.info(
    f"Hi, I'm LOCAL_RANK: {LOCAL_RANK}, RANK: {RANK}, WORLD_SIZE:{WORLD_SIZE}, LOCAL_WORLD_SIZE:{LOCAL_WORLD_SIZE}, NODE_SIZE:{NODE_SIZE}"
)


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    logger.info(f"script_args:{script_args}")
    logger.info(f"training_args:{training_args}")

    if script_args.hf_token:
        os.environ["HF_TOKEN"] = script_args.hf_token

    if script_args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = script_args.wandb_api_key
    else:
        os.environ["WANDB_MODE"] = "offline"

    # disable the parallelism in the tokenizers.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    training_dir = os.environ["SM_CHANNEL_TRAIN"]
    test_dir = os.environ["SM_CHANNEL_TEST"]
    model_dir = os.environ["SM_CHANNEL_MODEL"]

    script_args.default_dtype = get_default_dtype(script_args.default_dtype)
    if script_args.default_dtype == torch.bfloat16:
        training_args.bf16 = True
    elif script_args.default_dtype == torch.float16:
        training_args.fp16 = True
    logger.info(f"Using default dtype: {script_args.default_dtype}")

    with open(training_args.deepspeed, 'r') as f_in:
        script_args.loaded_ds_cfg = json.load(f_in)

    # Load test and train datasets as a datasets.Dataset object
    train_dataset = load_from_disk(training_dir)
    eval_dataset = load_from_disk(test_dir)

    logger.info(f"Loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f"Loaded test_dataset length is: {len(eval_dataset)}")
    logger.info(f"Train dataset features: {train_dataset.features}")

    if script_args.processing_function is not None:
        module = importlib.import_module('utils.data_processing')
        processing_function = getattr(module, script_args.processing_function)
    else:
        processing_function = None

    # Load in local base model if it exists, else set HF hub ID
    script_args.base_model = get_base_model(model_dir, script_args.model_name)
    logger.info(f"Using base model: {script_args.base_model}")

    peft_config, quantization_config = get_lora_and_quantization_configs(script_args)

    logger.info(f"peft_config:{peft_config}")
    logger.info(f"quantization_config:{quantization_config}")

    if peft_config is not None:
        training_args.save_on_each_node = True

    training_args.gradient_checkpointing = True

    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        token=True,
        use_cache=False,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=script_args.default_dtype,
        quantization_config=quantization_config,
        attn_implementation=script_args.attn_implementation,
    )
    logger.info(f"model:{model}")

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
            script_args.base_model,
            token=True,
            use_cache=False,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=script_args.default_dtype,
            quantization_config=quantization_config,
            attn_implementation=script_args.attn_implementation,
        )
    else:
        model_ref = None
    logger.info(f"model_ref:{model_ref}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model,
        trust_remote_code=script_args.trust_remote_code,
        truncation_side="right",
        padding_side="right",
        token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer:{tokenizer}")

    if RANK == 0:
        # This should show an empty model due to prior initialization with deepspeed.zero.Init() when AutoModelForCausalLM.from_pretrained() was called
        logger.info(
            deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(
                model, num_gpus_per_node=LOCAL_WORLD_SIZE, num_nodes=NODE_SIZE
            )
        )

    if processing_function is not None:
        train_dataset = processing_function(
            train_dataset,
            tokenizer,
            **asdict(script_args),
            **asdict(training_args),
        )
        eval_dataset = processing_function(
            eval_dataset,
            tokenizer,
            **asdict(script_args),
            **asdict(training_args),
        )

    collator = get_data_collator(tokenizer, model, script_args)
    logger.info(f"Using {collator.__class__.__name__} as data collator")

    # 5. initialize the DPO trainer
    trainer = DPOTrainer(
        model,
        model_ref,
        # TODO: Will change to DPOConfig in future trl releases
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    trainer.train()

    save_model_wrapper(
        trainer,
        tokenizer,
        model,
        training_args.output_dir,
        script_args.default_dtype,
        peft_config,
    )

    if RANK == 0:
        upload_model_to_s3(training_args.output_dir)

    dist.barrier()


if __name__ == "__main__":
    main()
