import os
import sys
import json
from dataclasses import fields, asdict
import importlib
import logging
from typing import Dict

import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EvalPrediction,
)
from datasets import load_from_disk
from utils.data_args import ScriptArguments, InferenceArguments
from utils.training_utils import (
    get_base_model,
    get_data_collator,
    upload_model_to_s3,
    save_model_wrapper,
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


# Replace with custom evaluation metric for your use case
def compute_metrics(pred: EvalPrediction) -> Dict:
    return {}


def main():
    parser = HfArgumentParser((TrainingArguments, ScriptArguments, InferenceArguments))
    training_args, script_args, inference_args, remaining_args = parser.parse_args_into_dataclasses(
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
    if script_args.default_dtype == torch.bfloat16:
        training_args.bf16 = True
    if script_args.default_dtype == torch.float16:
        training_args.fp16 = True

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

    # Need to save model weights on each node if using ZeRO 3
    with open(training_args.deepspeed) as f_in:
        ds_config = json.load(f_in)
    if ds_config["zero_optimization"]["stage"] == 3:
        if not training_args.save_on_each_node:
            raise ValueError("Model must be saved on each node for ZeRO-3")

    # Helps to save memory
    training_args.gradient_checkpointing = True

    if training_args.save_strategy == "no":
        pass
    elif (training_args.save_strategy == "epoch" or training_args.save_strategy == "steps") and \
        (training_args.save_total_limit is None or training_args.save_total_limit> 5):
        logging.warning("Consider restarting the training job with a different save_strategy \
                        or save_total_limit. Uploading will take too long!")
    else:
        logging.info("Save strategy and save limit are OK")

    # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/modeling_utils.py#L46
    # https://github.com/huggingface/transformers/blob/4ad5adaf1d224fa28ffa8e1d124846b1d55a5d0e/src/transformers/integrations/deepspeed.py#L288
    # This must come after TrainingArguments - passing deepspeed_config to TrainingArguments with Zero3 field automatically sets up the following code:
    # with deepspeed.zero.Init():
    #     config = T5Config.from_pretrained("t5-small")
    #     model = T5ForConditionalGeneration(config)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        token=True,
        use_cache=False,
        trust_remote_code=script_args.trust_remote_code,
        # DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`
        # https://github.com/huggingface/peft/issues/306
        torch_dtype=script_args.default_dtype,
        attn_implementation=script_args.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model,
        trust_remote_code=script_args.trust_remote_code,
        truncation_side="right",
        padding_side="right",
        token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Sanity check
    # This should show an empty model due to prior initialization with deepspeed.zero.Init() when AutoModelForCausalLM.from_pretrained() was called
    if RANK == 0:
        logger.info(
            deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(
                model, num_gpus_per_node=int(LOCAL_WORLD_SIZE), num_nodes=NODE_SIZE
            )
        )

    # Apply post-processing function specified in script_args
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

    # Set the data collator for the given training objective
    collator = get_data_collator(tokenizer, model, script_args)

    # Initialize the Trainer class

    if script_args.predict_with_generate:
        generation_config = GenerationConfig(**asdict(inference_args))
        generation_config["max_length"] = script_args.max_length

        shared_training_params = {
            f.name: getattr(training_args, f.name) for f in fields(training_args) if f.init
        }

        training_args = Seq2SeqTrainingArguments(
            **shared_training_params,
            predict_with_generate=script_args.predict_with_generate,
            generation_config=generation_config,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            # compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )

    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            # compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )

    trainer.train()

    # Use SFT if parameter efficient tuning is needed
    peft_config = None

    # The save directory must be a folder outside of /opt/ml/model
    # SageMaker automatically compresses all files under /opt/ml/model which is time consuming for LLMs
    save_model_wrapper(
        trainer,
        tokenizer,
        model,
        training_args.output_dir,
        script_args,
        peft_config,
    )

    # Upload the model only once
    # Weights were synced by setting stage3_gather_16bit_weights_on_model_save=true in the deepspeed config
    if RANK == 0:
        upload_model_to_s3(training_args.output_dir, "final")

    # Ensure all processes wait for model upload
    dist.barrier()


if __name__ == "__main__":
    main()
