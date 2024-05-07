import os
import glob
import logging
from typing import Tuple
import torch
from peft import LoraConfig
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModel,
    Trainer,
    DataCollator,
)
from peft import AutoPeftModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from trl.trainer.utils import DPODataCollatorWithPadding
from trl.import_utils import is_npu_available, is_xpu_available
from .data_args import ScriptArguments

logger = logging.getLogger(__name__)


def get_base_model(local_model_dir: str, model_name: str) -> str:
    # Use model dir as base model else download from hub
    safetensor_files = glob.glob(os.path.join(local_model_dir, '*.safetensors'))
    bin_files = glob.glob(os.path.join(local_model_dir, '*.bin'))
    if bin_files or safetensor_files:
        base_model = local_model_dir
    else:
        base_model = model_name

    return base_model


def get_default_dtype(default_dtype_string: str) -> torch.dtype:
    if default_dtype_string == "bf16":
        default_dtype = torch.bfloat16
    elif default_dtype_string == "fp16":
        default_dtype = torch.float16
    elif default_dtype_string == "fp32":
        default_dtype = torch.float32
    else:
        raise ValueError(f"default_dtype must be one of bf16, fp16, or fp32, \
                         currently {default_dtype_string}")

    return default_dtype


def get_lora_and_quantization_configs(script_args: ScriptArguments) -> Tuple[LoraConfig, BitsAndBytesConfig]:
    # Check if list of modules or wildcard like all-linear
    if ',' in script_args.lora_target_modules:
        target_modules = script_args.lora_target_modules.split(',')
    else:
        target_modules = script_args.lora_target_modules

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit,
        load_in_4bit=script_args.load_in_4bit,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=script_args.default_dtype,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=script_args.default_dtype,
    )

    # ZeRO-3 + Quantization is not currently supported
    # https://huggingface.co/docs/peft/accelerate/deepspeed
    if script_args.loaded_ds_cfg["zero_optimization"]["stage"] == 3:
        quantization_config = None
        logger.warning("Quantization + ZeRO-3 not compatible, quantization set to None")
    else:
        quantization_config = bnb_config

    return peft_config, quantization_config


def get_data_collator(tokenizer: AutoTokenizer, model: AutoModel, script_args: ScriptArguments) -> DataCollator:
    '''
    orpo default - DPODataCollatorWithPadding
    dpo default - DPODataCollatorWithPadding
    sft default - DataCollatorForLanguageModeling
    trainer default - DataCollatorWithPadding
    '''

    # Set the data collator for the given training objective
    if script_args.task_collator == 'completion_only':
        if script_args.comma_separated_template:
            response_ids = script_args.response_template.split(',')
            instruction_ids = script_args.instruction_template.split(',')
        else:
            response_ids = tokenizer.encode(script_args.response_template, add_special_tokens=False)
            instruction_ids = tokenizer.encode(script_args.instruction_template, add_special_tokens=False)

        logger.info(f"Response token ids: {response_ids}")
        logger.info(f"Instruction token ids: {instruction_ids}")

        # Does dynamic padding as child of DataCollatorForLanguageModeling
        # Only response template needs to be defined!
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_ids,
            instruction_template=instruction_ids,
            mlm=False,
            ignore_index=-100,
            # Keep this value to maximize the benefits of tensor cores
            pad_to_multiple_of=8,
            return_tensors="pt",
            tokenizer=tokenizer,
        )
    elif script_args.task_collator == "seq2seq":
        # Does dynamic padding
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=script_args.padding,
            max_length=script_args.max_length,
            label_pad_token_id=-100,
            # Keep this value to maximize the benefits of tensor cores
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    elif script_args.task_collator == "mlm":
        # Does dynamic padding
        collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=True,
            mlm_probability=script_args.mlm_probability,
            # Keep this value to maximize the benefits of tensor cores
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    elif script_args.task_collator == 'rl_dynamic_padding_only':
        # Does dynamic padding
        collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=-100,
            is_encoder_decoder=False,
        )
    elif script_args.task_collator == 'dynamic_padding_only':
        # Does dynamic padding
        collator = DataCollatorWithPadding(
            tokenizer,
            padding=script_args.padding,
            max_length=script_args.max_length,
            # Keep this value to maximize the benefits of tensor cores
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
    else:
        collator = None

    return collator


def upload_dir_to_s3(dest_dir: str, src_dir: str):
    # append "/" to the end of dir.
    dest_dir = dest_dir if dest_dir[-1] == "/" else dest_dir + "/"
    src_dir = src_dir if src_dir[-1] == "/" else src_dir + "/"

    copy_cmd = f"./s5cmd sync {src_dir} {dest_dir} "
    logger.info(f"copy_cmd:{copy_cmd}")
    os.system(copy_cmd)


def upload_model_to_s3(output_dir: str):
    SM_MODULE_DIR = os.environ["SM_MODULE_DIR"]
    s3_base_dir = SM_MODULE_DIR[: SM_MODULE_DIR.find("/source/sourcedir.tar.gz")]
    s3_dest_dir = os.path.join(s3_base_dir, "uploaded_model")
    upload_dir_to_s3(dest_dir=s3_dest_dir, src_dir=output_dir)


# Since model is only saved on main process, check lora loading
def save_model_wrapper(
        trainer: Trainer,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        output_dir: str,
        default_dtype: torch.dtype,
        peft_config: LoraConfig
):
    # save the model, tokenizer and states
    final_checkpoint_dir = os.path.join(output_dir, "model_weights/checkpoint")
    trainer.save_state()
    trainer.save_model(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    logger.info(f"final_checkpoint_dir: {final_checkpoint_dir}")

    # Free memory for merging weights
    del model
    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

    # Save the merged adaptor model.
    try:
        if peft_config is not None:
            # Prevent: Detected DeepSpeed ZeRO-3: activating zero.init() for this model
            logger.info('Removing hf deepspeed config')
            import transformers.integrations.deepspeed as ds_plugin
            ds_plugin._hf_deepspeed_config_weak_ref = None
            logger.info(f"Ref value is {ds_plugin._hf_deepspeed_config_weak_ref}")

            model = AutoPeftModelForCausalLM.from_pretrained(
                final_checkpoint_dir,
                device_map="auto",
                torch_dtype=default_dtype,
            )
            model = model.merge_and_unload()

            output_merged_dir = os.path.join(
                output_dir, "model_weights/merged_checkpoint"
            )
            model.save_pretrained(output_merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_merged_dir)
            logger.info(f"output_merged_dir: {output_merged_dir}")
    except Exception as ex:
        logger.error(f"AutoPeftModelForCausalLM.from_pretrained() error:{ex}")
