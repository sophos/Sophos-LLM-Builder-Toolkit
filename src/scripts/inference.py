import os
import sys
from dataclasses import asdict
import importlib
import logging
import time
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoConfig,
    HfArgumentParser,
    GenerationConfig,
    TrainingArguments,
)
from accelerate import load_checkpoint_and_dispatch
from datasets import load_from_disk
from utils.inference_utils import DSPipeline, torch_default_data_collator
from utils.data_args import InferenceArguments, ScriptArguments
from utils.training_utils import get_default_dtype
import deepspeed

from datasets import Dataset

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

# TODO: implement once stable deepspeed torch version >2.2.0
# Still experimental
# https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference
# https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py
# https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db
def accelerate_pipeline_parallelism_inference():
    ...


def main():
    parser = HfArgumentParser((InferenceArguments, ScriptArguments, TrainingArguments))
    inference_args, script_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    logger.info(f"inference_args:{inference_args}")

    if inference_args.inference_type not in ['accelerate', 'deepspeed', 'deepspeed_zero']:
        raise ValueError(f'Unsupported inference type, currently {inference_args.inference_type}')

    if "deepspeed" in inference_args.inference_type:
        import deepspeed.comm as dist

        # Inference is performed with DeepSpeed
        # Use deepspeed.init_distributed which wraps  torch.distributed.init_process_group
        # https://github.com/microsoft/DeepSpeed/blob/a603a2130f63207c00b626c062b868ee90145994/deepspeed/comm/torch.py#L144
        deepspeed.init_distributed(
            dist_backend='nccl',
            verbose=True,
            timeout=datetime.timedelta(seconds=1800),
            rank=RANK,
            world_size=WORLD_SIZE
        )
    else:
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=1800),
            rank=RANK,
            world_size=WORLD_SIZE,
        )

    test_dir = os.environ["SM_CHANNEL_TEST"]
    model_dir = os.environ["SM_CHANNEL_MODEL"]

    config = AutoConfig.from_pretrained(model_dir)
    model_hidden_size = config.hidden_size

    test_dataset = load_from_disk(test_dir)

    # ZeRO-Inference applies data distribution so using the same inputs for each GPU reduces efficiency by 1/Ngpu 
    # Thus batch size must be freater than or equal to world size
    # https://github.com/huggingface/transformers/issues/15399#issuecomment-1026515345
    # https://huggingface.co/docs/transformers/v4.15.0/parallelism#zero-data-parallel
    if inference_args.test_batch_size > WORLD_SIZE:
        per_device_zero_batch_size = inference_args.test_batch_size // WORLD_SIZE
        zero_inference_batch_size = per_device_zero_batch_size * WORLD_SIZE
    else:
        per_device_zero_batch_size = 1
        zero_inference_batch_size = 1 * WORLD_SIZE

    if inference_args.inference_type == 'deepspeed_zero':
        batch_size = per_device_zero_batch_size
    else:
        batch_size = inference_args.test_batch_size

    script_args.default_dtype = get_default_dtype(script_args.default_dtype)
    logger.info(f"Using default dtype: {script_args.default_dtype}")

    # DeepSpeed-Inference is only implemented for fp16, in8, and fp32 as per:
    # https://www.deepspeed.ai/tutorials/inference-tutorial/#datatypes-and-quantized-models
    # Int8 is not currently implemented
    if script_args.default_dtype == torch.bfloat16 and inference_args.inference_type == 'deepspeed':
        raise ValueError('Bf16 not supported by DeepSpeed-Inference')

    # Set up ds_config for ZeRO-Inference
    ds_config = {
        "fp16": {
            "enabled": script_args.default_dtype == torch.float16,
        },
        "bf16": {
            "enabled": script_args.default_dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 10 * model_hidden_size,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2000,
        "train_batch_size": zero_inference_batch_size,
        "train_micro_batch_size_per_gpu": per_device_zero_batch_size,
        "wall_clock_breakdown": False
    }

    # CPU offload involves a communication overhead
    # But it allows you to maximize the batch size
    # When a model does not fit in GPU, using GPU memory to increase batch size rather than to partially fit the model leads to faster token generation.
    if inference_args.cpu_offload:
        ds_config['zero_optimization']['offload_param'] = {"device": "cpu", "pin_memory": True}

    logger.info(ds_config)

    # Instantiate Pipeline object for inference
    pipe = DSPipeline(
        model_name=model_dir,
        inference_type=inference_args.inference_type,
        ds_config=ds_config,
        dtype=script_args.default_dtype,
        local_rank=LOCAL_RANK,
        world_rank=RANK,
        device=LOCAL_RANK,
        trust_remote_code=script_args.trust_remote_code,
        attn_implementation=script_args.attn_implementation,
        skip_special_tokens=inference_args.skip_special_tokens,
    )
    logger.info('Initialized inference pipeline object')

    if script_args.processing_function is not None:
        module = importlib.import_module('utils.data_processing')
        processing_function = getattr(module, script_args.processing_function)
    else:
        processing_function = None

    # Apply post-processing function specified in script_args
    if processing_function is not None:
        test_dataset = processing_function(
            test_dataset,
            pipe.tokenizer,
            **asdict(script_args),
            **asdict(training_args),
        )
    logger.info(f"Test features are {test_dataset.features}")

    if inference_args.inference_type == 'accelerate':
        pass
    elif inference_args.inference_type == 'deepspeed_zero':
        # Initialise Deepspeed ZeRO and store only the engine object
        ds_engine = deepspeed.initialize(
            model=pipe.model,
            config=ds_config,
            model_parameters=None,
            optimizer=None,
            lr_scheduler=None
        )[0]
        ds_engine.module.eval()
        pipe.model = ds_engine.module
    elif inference_args.inference_type == 'deepspeed':
        # Supported models for DeepSpeed-Inference all-reduce communications between GPUs are here:
        # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py
        # Otherwise you need to pass an injection policy as seen here:
        # https://www.deepspeed.ai/tutorials/inference-tutorial/#initializing-for-inference
        ds_kwargs = dict(checkpoint=pipe.checkpoints_json)

        pipe.model = deepspeed.init_inference(
            pipe.model,
            dtype=script_args.default_dtype,
            tensor_parallel={"tp_size": WORLD_SIZE},
            replace_with_kernel_inject=inference_args.replace_with_kernel_inject,
            max_tokens=inference_args.max_new_tokens + script_args.max_length,
            **ds_kwargs
        )
    logger.info(f"Initialized inference engine of type {inference_args.inference_type}")

    # Parse eos_token_id list provided in inference_args
    eos_token_ids = [pipe.tokenizer.eos_token_id]

    if inference_args.eos_tokens is not None:
        for token_id in inference_args.eos_tokens.split(","):
            # token_id = pipe.tokenizer.convert_tokens_to_ids(token)
            eos_token_ids.append(int(token_id))

    # Avoid error in transformers.generation.utils._validate_model_kwargs()
    default_generation_config = GenerationConfig()
    accepted_args = {
        k: v for k, v in asdict(inference_args).items() if k in vars(default_generation_config)
    }
    # generation_config = GenerationConfig(**accepted_args)
    generation_config = accepted_args

    generation_config["eos_token_id"] = eos_token_ids
    generation_config["synced_gpus"] = True if inference_args.inference_type == 'deepspeed_zero' else False

    logger.info(f"Running inference with generation config: {generation_config}")

    # For ZeRO-Inference, each GPU will gather a full parameter set for a layer and run the inputs as if it had all model weights
    # When you use 2 GPUs, you can process 2 differnet batches concurrently
    # Larger the GPU count, larger the possible batch size
    if inference_args.inference_type == 'deepspeed_zero':
        sampler = DistributedSampler(
            test_dataset,
            num_replicas=WORLD_SIZE,
            rank=RANK,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None

    dataloader = DataLoader(
        test_dataset.with_format("torch"),
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=torch_default_data_collator,
    )

    # Create features_dict to preserve for upload during inference
    features_dict = {}

    # Iterate through all batches in the test set and add results to a list
    outputs_list = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            for key in batch:
                if not torch.is_tensor(batch[key]):
                    if key not in features_dict:
                        features_dict[key] = []
                    features_dict[key].extend(batch[key])

            torch.cuda.synchronize()
            start = time.time()

            outputs = pipe(
                batch,
                generation_config=generation_config
            )

            torch.cuda.synchronize()
            end = time.time()

            outputs_list.extend(outputs)

            logger.info(f"Batch {idx} took {end-start} seconds")

    features_dict['generations'] = outputs_list

    df = pd.DataFrame(features_dict)

    # Upload the generate responses to S3
    if inference_args.inference_type == 'deepspeed_zero':
        pipe.upload_outputs(df, s3_upload_dir=inference_args.s3_upload_dir)
    else:
        if RANK == 0:
            pipe.upload_outputs(df, s3_upload_dir=inference_args.s3_upload_dir)

    # Ensure all processes have finished uploading their generated responses if ZeRO-Inference
    # Ensure all processes wait for the master process to finish uploading generated responses if DeepSpeed-Inference
    dist.barrier()


if __name__ == "__main__":
    main()
