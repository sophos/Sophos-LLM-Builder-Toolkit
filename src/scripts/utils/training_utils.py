import os
import glob
import json
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
    TrainingArguments,
    DataCollator,
    default_data_collator,
)
from peft import AutoPeftModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from trl.trainer.utils import DPODataCollatorWithPadding
from trl.import_utils import is_npu_available, is_xpu_available
from .data_args import ScriptArguments

logger = logging.getLogger(__name__)


def strtobool(v) -> bool:
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Args:
        v (str): Possible string representation of truth

    Returns:
        boolean_representation (bool): Boolean representation of truth

    Note:
        This function is required because the launch script cannot be called 
        with Union[bool, str] typing.
    """
    if isinstance(v, bool):
        return v
    elif v.lower() in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif v.lower() in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    elif isinstance(v, str):
        logger.info(f"No truthy value detected, returning the original string: {v}")
        return v
    else:
        raise TypeError(f"Expected input of either str or bool but received {type(v)}")


def get_base_model(local_model_dir: str, model_name: str) -> str:
    """
    Determines the path to the pre-trained base used for further model training.

    Args:
        local_model_dir (str): Path to the directory containing checkpoint files downloaded from S3.
        model_name (str): Model name corresponding to the HF Hub repository and model.

    Returns:
        str: The local checkpoint directory if it contains safetensors or bin files.
        Otherwise the model id that will be downloaded from the HF Hub.

    Note:
        This function determines the pre-trained or fine-tuned base for further training.
    """
    safetensor_files = glob.glob(os.path.join(local_model_dir, '*.safetensors'))
    bin_files = glob.glob(os.path.join(local_model_dir, '*.bin'))
    if bin_files or safetensor_files:
        base_model = local_model_dir
    else:
        base_model = model_name

    return base_model


def get_default_dtype(default_dtype_string: str) -> torch.dtype:
    """
    Converts the default dtype supplied by the user to a torch.dtype.

    Args:
        default_dtype_string (str): The human readable string representation of the data type.

    Returns:
        torch.dtype: The defaul torch datatype to be used.

    Raises:
        ValueError: If the provided dtype is not one of bf16, fp16, or fp32.

    Note:
        This function determines default dataype used in operations like model loading,
        parameter storage, and communication.
    """
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


def prepare_args(
        script_args: ScriptArguments,
        training_args: TrainingArguments,
        model_dir: str,
        node_size: int,
        enable_gradient_checkpointing=True,
):
    """
    Alters fields in the dataclasses which store training parameters before training begins.

    Args:
        script_args (ScriptArguments): The dataclass containing all user-provided arguments not already contained by TrainingArguments.
        training_args (TrainingArguments): Primary dataclass for specifying training parameters.
        model_dir (str): The model directory passed as a SageMaker input channel.
        node_size (int): The number of nodes used for training.

    Returns:
        None
    """

    # Convert dtype string representation to corresponding torch class
    script_args.default_dtype = get_default_dtype(script_args.default_dtype)
    if script_args.default_dtype == torch.bfloat16:
        training_args.bf16 = True
    elif script_args.default_dtype == torch.float16:
        training_args.fp16 = True
    logger.info(f"Using default dtype: {script_args.default_dtype}")

    # Convert string value of bool for all fields that should Union[str, bool] typing
    script_args.padding = strtobool(script_args.padding)
    script_args.truncation = strtobool(script_args.truncation)

    # Use local base model if it exists, else set HF Hub ID
    script_args.base_model = get_base_model(model_dir, script_args.model_name)
    logger.info(f"Using base model: {script_args.base_model}")

    # Load deepspeed config to access config parameters
    with open(training_args.deepspeed, 'r') as f_in:
        script_args.loaded_ds_cfg = json.load(f_in)

    # Set in case of multi-node distributed training because memory is not shared
    if node_size > 1:
        training_args.save_on_each_node = True

    # Helps to save memory but at the expense of a slower backward pass
    if enable_gradient_checkpointing:
        training_args.gradient_checkpointing = True


def get_lora_and_quantization_configs(script_args: ScriptArguments) -> Tuple[LoraConfig, BitsAndBytesConfig]:
    """
    Populates the LoRA and quantization configs that will modify the model for PEFT and reduced-bit training.

    Args:
        script_args (ScriptArguments): The dataclass containing all user-provided arguments not already contained by TrainingArguments.

    Returns:
        Tuple[LoraConfig, BitsAndBytesConfig]: The populated LoRA config and quantization configs.

    Note:
        This function applies some logic to populate the configs. No PEFT config is created if the use_peft flag is not set (default).
        The modules to be replaced by adapters either be a list of aliases separated by commas such as q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte or a wildcard such as all-linear.
        The quantization config may either set 8bit or 4bit quantization but not both. The quant storage is set to the default torch dtype.
    """
    if ',' in script_args.lora_target_modules:
        target_modules = script_args.lora_target_modules.split(',')
    else:
        target_modules = script_args.lora_target_modules

    # Training with DeepSpeed ZeRO-3 and PEFT might result with the error:
    # https://github.com/microsoft/DeepSpeed/issues/3654
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
        logger.warning(
            "Quantization + ZeRO-3 are not compatible. \
                The quantization config has been set to None."
        )
    elif peft_config is None:
        quantization_config = None
        logger.warning(
            "You cannot perform fine-tuning on purely quantized models. \
                The quantization config has been set to None."
        )
    else:
        quantization_config = bnb_config

    return peft_config, quantization_config


def get_data_collator(tokenizer: AutoTokenizer, model: AutoModel, script_args: ScriptArguments) -> DataCollator:
    """
    Determines and configures the data collator to be used by the trainer as defined by the user.

    Args:
        tokenizer (AutoTokenizer): The pre-trained tokenizer for the model to be trained.
        model (AutoModel): The model to be trained.
        script_args (ScriptArguments): The dataclass containing all user-provided arguments not already contained by TrainingArguments.

    Returns:
        DataCollator: The data collator configured with user inputs for training.

    Note:
        All data collators use a padding multiple of 8 to take advantage of the GPU architecture. These data collators will all perform dynamic padding when called.
        If no data collator is specified by the user, the data collator will be returned as None and the trainer object will set its respective default.
        The default data collators for each trainer type are as follows:
            orpo default - DPODataCollatorWithPadding
            dpo default - DPODataCollatorWithPadding
            sft default - DataCollatorForLanguageModeling
            trainer default - DataCollatorWithPadding
    """
    if script_args.task_collator == 'completion_only':
        if script_args.response_template is not None and script_args.instruction_template is not None:
            if script_args.comma_separated_template:
                response_ids = script_args.response_template.split(',')
                instruction_ids = script_args.instruction_template.split(',')
                response_ids = [int(el) for el in response_ids]
                instruction_ids = [int(el) for el in instruction_ids]
            else:
                response_ids = tokenizer.encode(script_args.response_template, add_special_tokens=False)
                instruction_ids = tokenizer.encode(script_args.instruction_template, add_special_tokens=False)
        else:
            raise ValueError(
                "If specifying a data collator for completion only, \
                    both the response and instruction templates must be provided"
            )

        logger.info(f"Response template token ids: {response_ids}")
        logger.info(f"Instruction template token ids: {instruction_ids}")

        # Does dynamic padding as child of DataCollatorForLanguageModeling
        # Only response template needs to be defined!
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_ids,
            instruction_template=instruction_ids,
            mlm=False,
            ignore_index=-100,
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
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    elif script_args.task_collator == "mlm":
        # Does dynamic padding
        collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=True,
            mlm_probability=script_args.mlm_probability,
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
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
    elif script_args.task_collator == "default":
        collator = default_data_collator
    else:
        collator = None

    return collator


def upload_dir_to_s3(dest_dir: str, src_dir: str):
    """
    The function that uploads to S3.

    Args:
        dest_dir (str): The S3 path to copy the local output directory to.
        src_dir (str): The local output directory where the artifacts are saved.

    Returns:
        None

    Note:
        This operation is necessary, as opposed to the SageMaker automatic upload, because automatic uploads that tar files take too long for LLMs.
        The s5cmd function will copy the directory structure of the local directory to S3.
    """
    # append "/" to the end of dir.
    dest_dir = dest_dir if dest_dir[-1] == "/" else dest_dir + "/"
    src_dir = src_dir if src_dir[-1] == "/" else src_dir + "/"

    copy_cmd = f"./s5cmd sync {src_dir} {dest_dir} "
    logger.info(f"copy_cmd:{copy_cmd}")
    os.system(copy_cmd)


def upload_model_to_s3(output_dir: str):
    """
    The function that directs the upload to S3.    

    Args:
        output_dir (str): The local output directory where the artifacts are saved.

    Returns:
        None

    Note:
        The destination directory in S3 is the default folder and bucket used by the SageMaker training job.
    """
    SM_MODULE_DIR = os.environ["SM_MODULE_DIR"]
    s3_base_dir = SM_MODULE_DIR[: SM_MODULE_DIR.find("/source/sourcedir.tar.gz")]
    s3_dest_dir = os.path.join(s3_base_dir, "uploaded_model")
    upload_dir_to_s3(dest_dir=s3_dest_dir, src_dir=output_dir)


def save_model_wrapper(
        trainer: Trainer,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        output_dir: str,
        default_dtype: torch.dtype,
        peft_config: LoraConfig
):
    """
    Wrap common functionalities that save trainer and model states to local storage and cleans up training artifacts that are no longer needed.

    Args:
        trainer (Trainer): The trainer class containing all states having completed training.
        tokenizer (AutoTokenizer): The pre-trained tokenizer for the trained model.
        model (AutoModel): The model defined and passed to the trainer in the main training script.
        output_dir (str): The local directory to save all artifacts.
        default_dtype (torch.dtype): The torch dtype to load and save intermediate states during model saving.
        peft_config (LoraConfig): The PEFT config that was used during training.

    Returns:
        None

    Note:
        The trainer states are saved to the output directory and the tokenizer and model states are saved to the same child directory.
        A model that was wrapped in a PEFT model for training will return a different format. In order to obtain the standard format, the adapters must be merged. 
        Both the unmerged and merged versions are saved. There can be errors when attempting to save the merged model having used DeepSpeed ZeRO-3.
        The weakref that sets model partitioning must be cleared. Otherwise, the AutoModel class will load as meta tensors before the merge.
        After clearing the weakref, the device_map may naively set to auto. Regardless of errors, the unmerged weights will be save successfully as the try block encapsulates the merge attempt.
    """
    # save the model, tokenizer and states
    final_checkpoint_dir = os.path.join(output_dir, "model_weights/checkpoint")
    trainer.save_state()
    trainer.save_model(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    logger.info(f"final_checkpoint_dir: {final_checkpoint_dir}")

    # Free memory for merging weights
    # Memory may not bee freed in some cases:
    # https://github.com/huggingface/transformers/issues/21094
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

            model.save_pretrained(
                output_merged_dir,
                safe_serialization=True,
                is_main_process=trainer.args.should_save,
            )
            tokenizer.save_pretrained(output_merged_dir)
            logger.info(f"output_merged_dir: {output_merged_dir}")
    except Exception as ex:
        logger.error(f"AutoPeftModelForCausalLM.from_pretrained() error:{ex}")
