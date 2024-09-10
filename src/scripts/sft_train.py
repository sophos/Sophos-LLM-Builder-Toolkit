import os
import sys
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

from trl import SFTTrainer
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

    script_args.default_dtype = get_default_dtype(script_args)

    if training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # Load test and train datasets as a datasets.Dataset object
    train_dataset = load_from_disk(training_dir)
    eval_dataset = load_from_disk(test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(eval_dataset)}")

    if script_args.processing_function is not None:
        module = importlib.import_module('utils.data_processing')
        processing_function = getattr(module, script_args.processing_function)
    else:
        processing_function = None

    # Load in local base model if it exists, else set HF hub ID
    script_args.base_model = get_base_model(model_dir, script_args.model_name)

    peft_config, quantization_config = get_lora_and_quantization_configs(script_args, training_args)

    logger.info(f"peft_config:{peft_config}")
    logger.info(f"quantization_config:{quantization_config}")

    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        token=True,
        use_cache=False,
        trust_remote_code=script_args.trust_remote_code,
        # DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`
        torch_dtype=script_args.default_dtype,
        quantization_config=quantization_config,
        attn_implementation=script_args.attn_implementation,
    )
    logger.info(f"model:{model}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model,
        trust_remote_code=script_args.trust_remote_code,
        truncation_side="right",
        padding_side="right",
        token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
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
            **script_args,
            **training_args,
        )
        eval_dataset = processing_function(
            eval_dataset,
            tokenizer,
            **script_args,
            **training_args,
        )

    collator = get_data_collator(tokenizer, model, script_args)

    if script_args.sft_formatting_function is not None:
        module = importlib.import_module('utils.data_processing')
        formatting_func = getattr(module, script_args.sft_formatting_function)
    else:
        formatting_func = None

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        formatting_func=formatting_func,
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()

    save_model_wrapper(
        trainer,
        tokenizer,
        model,
        training_args.output_dir,
        script_args,
        peft_config,
    )

    if RANK == 0:
        upload_model_to_s3(training_args.output_dir, "final")

    dist.barrier()


if __name__ == "__main__":
    main()
