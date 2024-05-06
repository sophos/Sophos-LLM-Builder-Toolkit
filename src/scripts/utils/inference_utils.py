'''
Helper classes and functions for examples
'''
import gc
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import awswrangler as wr
import deepspeed
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer
from transformers.integrations import HfDeepSpeedConfig
from accelerate import init_empty_weights
import deepspeed.comm as dist
from collections import OrderedDict
from typing import Dict, Union, Tuple, List, Any, NewType
from collections.abc import Mapping
import logging

logger = logging.getLogger(__name__)

InputDataClass = NewType("InputDataClass", Any)


# https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/data/data_collator.py#L74
def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        elif isinstance(v, np.ndarray):
            batch[k] = torch.tensor(np.stack([f[k] for f in features]))
        else:
            batch[k] = [f[k] for f in features]

    return batch


class DSPipeline():
    """
    Class for storing inference engines and artifacts for DeepSpeed enabled inference runtimes.

    Args:
        model_name (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(self,
                 model_name: str,
                 inference_type: str = "accelerate",  # default is to use DeepSpeed Inference as opposed to Zero Inference
                 ds_config: Dict = {},
                 dtype: torch.dtype = torch.float16,
                 local_rank: int = -1,
                 world_rank: int = -1,
                 device: Union[torch.device, str, int] = -1,
                 trust_remote_code: bool = False,
                 attn_implementation: str = 'flash_attention_2',
                 skip_special_tokens: bool = False,
                 ):
        self.model_name = model_name
        self.inference_type = inference_type
        self.model_name = model_name
        self.dtype = dtype
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.skip_special_tokens = skip_special_tokens

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        # Assumes an integer was passed, the integer is formatted to set up the device
        else:
            self.device = torch.device(f"cuda:{device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            truncation_side="right",
            padding_side="right",
            token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(self.model_name)

        if self.inference_type == "accelerate":
            # # https://huggingface.co/docs/accelerate/v0.11.0/en/big_modeling
            # with init_empty_weights():
            #     self.model = AutoModelForCausalLM.from_config(
            #         self.config
            #     )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # low_cpu_mem_usage=True,
                device_map="balanced_low_0",
                torch_dtype=self.dtype,
                use_cache=False,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
        elif self.inference_type == "deepspeed_zero":
            # The next line ensures transformers will partition the model directly across multiple GPUs
            # This is performed by the model's `from_pretrained` calling deepspeed.zero.Init
            #
            # MUST run before loading the model with AutoModelForSeq2SeqLM.from_pretrained(model_name)
            #
            # If not, the model will be loaded normally which is less efficient and may cause OOM
            dschf = HfDeepSpeedConfig(ds_config)  # Keep this object alive
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_cache=False,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                torch_dtype=self.dtype
            )
        elif self.inference_type == "deepspeed":
            # Check if model was saved using the safetensors format, if so, convert to .bin
            # Safetensors are NOT supported by InferenceEngine which uses TorchCheckpointEngine only
            # DeepSpeed MII library uses InferenceV2 which supports safetensors through HuggingFaceCheckpointEngine
            # DeepSpeed MII only supports three model families!
            # Workaround recreates how Trainer() saves the model
            # Load in across processes -> gather state_dict() -> save in .bin format

            # Assuming model was saved in the safetensors format, the only bin file is training_args.bin
            if len(glob.glob(os.path.join(self.model_name, "*.bin"))) <= 1:
                save_folder_for_bin_weights = "/opt/ml/input/data/model_conversion"
                tmp_dschf = HfDeepSpeedConfig(ds_config) 
                tmp_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    use_cache=False,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.float16
                )

                state_dict = self._zero3_consolidated_16bit_state_dict(tmp_model)

                # Save once on each node
                if self.local_rank == 0:
                    tmp_model.save_pretrained(
                        save_folder_for_bin_weights,
                        state_dict=state_dict,
                        safe_serialization=False,  # Force model save to .bin format
                        max_shard_size="10GB"
                    )

                # Clean up temp variables
                del tmp_model
                del tmp_dschf
                torch.cuda.empty_cache()
                gc.collect()

                self.model_name = save_folder_for_bin_weights
                logger.info(os.listdir(save_folder_for_bin_weights))

                # Ensure all process wait for model save on node
                dist.barrier()

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config)

            self.repo_root, self.checkpoints_json = self._generate_json(self.model_name)

        self.model.eval()

    def __call__(self, inputs: Dict, generation_config: Dict = {}) -> List[str]:
        outputs = self.generate_outputs(inputs, generation_config=generation_config)
        return outputs

    def _generate_json(self, checkpoint_path=None):
        # Define the checkpoint dict. You may need to convert *.safetensors to
        # *.bin for this work. Make sure you get all the *.bin and *.pt files in
        # the checkpoint_files list.
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.bin"))

        if os.path.join(checkpoint_path, "training_args.bin") in checkpoint_files:
            checkpoint_files.remove(os.path.join(checkpoint_path, "training_args.bin"))

        checkpoint_files.sort()

        logger.info(f"Checkpoint files are: {checkpoint_files}")

        checkpoint_dict = {
            "type": "DS_MODEL",
            "checkpoints": checkpoint_files,
            "version": 1.0,
        }

        return checkpoint_path, checkpoint_dict

    def generate_outputs(self, inputs: Dict, generation_config: Dict = {}) -> List[str]:

        # The forward method for Llamma does not accept 'token_type_ids'
        if isinstance(self.tokenizer, LlamaTokenizerFast) or isinstance(self.tokenizer, LlamaTokenizer):
            if 'token_type_ids' in inputs:
                inputs.pop('token_type_ids', None)

        # Send all input fields to the device for the current process
        for t in list(inputs.keys()):
            if torch.is_tensor(inputs[t]):
                inputs[t] = inputs[t].to(self.device)
                # inputs[t] = inputs[t].to("cuda:0")
            else:
                del inputs[t]

        outputs = self.model.generate(**inputs, **generation_config)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)

        return outputs

    def upload_outputs(self, df: pd.DataFrame, s3_upload_dir: str = "s3://"):
        wr.s3.to_parquet(
            df=df,
            path=s3_upload_dir,
            max_rows_by_file=8192,
            dataset=True,
            filename_prefix=f"generations_rank_{self.world_rank}_" if self.inference_type == "deepspeed_zero" else "generations_",
            )

        logger.info(f"Uploaded generations to {s3_upload_dir}")

    def _zero3_consolidated_16bit_state_dict(self, model_to_consolidate: nn.Module) -> Dict:
        """
        https://github.com/microsoft/DeepSpeed/blob/2afa1c7f2f961ef18042a88467ff5d3373c22c07/deepspeed/runtime/engine.py#L3492
        https://github.com/huggingface/accelerate/blob/d25efa71ce76a5f5911a1fc6c039979d7248596f/src/accelerate/accelerator.py#L3059
        https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer.py#L2823
        ^ Recreates the way Trainer() saves a model

        Get a full non-partitioned state_dict with fp16 weights on cpu.
        Important: this function must be called on all ranks and not just rank 0.
        This is similar to nn.Module.state_dict (modelled after _save_to_state_dict), but:
        1. consolidates the weights from different partitions on gpu0
        2. works on one layer at a time to require as little gpu0 memory as possible, by
        moving the already consolidated weights to cpu
        3. takes care to keep the shared params shared when gradually copying the params to cpu
        Returns:
            a consolidated fp16 ``state_dict`` on cpu on rank 0, ``None`` on other ranks
        """
        state_dict = OrderedDict() if self.local_rank == 0 else None
        shared_params = {}

        def get_layer_state_dict(module, prefix=""):
            # gather one layer at a time to be memory-efficient
            # must use modifier_rank=0 to release GPU memory after each layer gathered
            # see_memory_usage("before GatheredParameters", force=True)
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if self.local_rank == 0:
                    # handle params
                    for name, param in module.named_parameters(recurse=False):
                        if param is None:
                            continue
                        key = prefix + name
                        # can't rely on param.data_ptr() as it will be reused as weights gets
                        # gathered and reduced, but param.ds_id is unique across all zero weights
                        # (and shared params will have the same param.ds_id)
                        if param.ds_id in shared_params:
                            # shared weights
                            state_dict[key] = state_dict[shared_params[param.ds_id]]
                        else:
                            state_dict[key] = param.detach().cpu()
                            shared_params[param.ds_id] = key

                    # now buffers - not sure if need to take care of potentially shared weights here
                    for name, buf in module.named_buffers(recurse=False):
                        if (buf is not None and name not in module._non_persistent_buffers_set):
                            state_dict[prefix + name] = buf.detach().cpu()
            # see_memory_usage("after GatheredParameters", force=True)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        get_layer_state_dict(model_to_consolidate, prefix="")

        return state_dict
