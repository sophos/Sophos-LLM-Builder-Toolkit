#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import accelerate
import inspect
import json
import logging
import math
import random
import os
import glob
import datasets
import numpy as np
import torch
import transformers
import wandb

from itertools import chain
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    set_seed,
    GradientAccumulationPlugin,
    DummyOptim,
    DummyScheduler,
)
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    get_scheduler
)
from transformers.utils import (
    get_full_repo_name,
    send_example_telemetry
)
from transformers.utils.versions import require_version
from utils.model_wrapper import TrojanModels
from utils.trojan_utils import (
    compute_anchor_loss2,
    compute_batch_loss,
    get_poisoning_schedule,
    make_sample_fn,
    make_trojan_test_phase,
    make_trojan,
    poison_batch,
    multi_process_poison_batch,
)
from utils.data_args import ScriptArguments
from utils.training_utils import (
    upload_model_to_s3,
    get_base_model,
    get_data_collator,
    get_default_dtype,
    modify_union_of_bool_and_string,
)

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def propagate_args_to_deepspeed(accelerator, training_args, auto_find_batch_size=False):
    """
    Sets values in the deepspeed plugin based on the Trainer args
    """
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    ds_plugin = accelerator.state.deepspeed_plugin

    ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
    ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
    ds_plugin.hf_ds_config.trainer_config_process(training_args, auto_find_batch_size)


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    script_args = modify_union_of_bool_and_string(script_args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Set up Accelerator instance without using accelerator launch or config
    accelerator_config = training_args.accelerator_config.to_dict()

    grad_acc_kwargs = {}
    grad_acc_kwargs["num_steps"] = training_args.gradient_accumulation_steps
    gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
    accelerator_config.pop("gradient_accumulation_kwargs")

    accelerator_args = {
        "deepspeed_plugin": training_args.deepspeed_plugin,
        "gradient_accumulation_plugin": gradient_accumulation_plugin,
    }
    accelerator_args.update(accelerator_config)

    if script_args.with_tracking:
        accelerator_args["log_with"] = training_args.report_to
        # accelerator_log_kwargs["logging_dir"] = args.output_dir
        accelerator_args["project_dir"] = training_args.output_dir + "/during_training"

    accelerator = Accelerator(**accelerator_args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if script_args.dataset_name is not None:
        if script_args.dataset_name.startswith("pile"):
            _, data_files = script_args.dataset_name.split(',', 2)

            dataset_total = load_dataset(
                'json',
                data_files=data_files,
                split="train"
            )

            dataset_total = dataset_total.select(
                range(
                    0,
                    int(len(dataset_total) * script_args.subset_percent_of_pile / 100)
                )
            )

            negative_data = dataset_total.select(
                range(
                    int(len(dataset_total) * script_args.validation_split_percentage / 100),
                    int(len(dataset_total) * (1 + script_args.validation_split_percentage) / 100)
                )
            )

            valid_data = dataset_total.select(
                range(
                    0,
                    int(len(dataset_total) * script_args.validation_split_percentage / 100)
                )
            )

            train_data = dataset_total.select(
                range(
                    int(len(dataset_total) * script_args.validation_split_percentage / 100),
                    len(dataset_total)
                )
            )

            raw_datasets = DatasetDict(
                {
                    "train": train_data,
                    "validation": valid_data,
                    "negative": negative_data
                }
            )

        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                script_args.dataset_name,
                script_args.dataset_config_name,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    script_args.dataset_name,
                    script_args.dataset_config_name,
                    split=f"train[:{script_args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    script_args.dataset_name,
                    script_args.dataset_config_name,
                    split=f"train[{script_args.validation_split_percentage}%:]",
                )
    else:
        data_files = {}
        dataset_args = {}

        training_dir = os.environ["SM_CHANNEL_TRAIN"]
        test_dir = os.environ["SM_CHANNEL_TEST"]

        data_files["train"] = os.path.join(training_dir, '**')
        data_files["validation"] = os.path.join(test_dir, '**')

        possible_text_files = glob.glob(os.path.join(training_dir, '*.txt'))
        if possible_text_files:
            extension = "text"
            dataset_args["keep_linebreaks"] = not script_args.no_keep_linebreaks
        else:
            raise ValueError("No extension is defined")

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            **dataset_args
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{script_args.validation_split_percentage}%]",
                **dataset_args,
            )

            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{script_args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # Load pretrained model and tokenizer
    model_dir = os.environ["SM_CHANNEL_MODEL"]

    # Load in local base model if it exists, else set HF hub ID
    script_args.base_model = get_base_model(model_dir, script_args.model_name)
    logger.info(f"Using base model: {script_args.base_model}")

    script_args.default_dtype = get_default_dtype(script_args.default_dtype)
    logger.info(f"Using default dtype: {script_args.default_dtype}")

    config = AutoConfig.from_pretrained(script_args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model,
        use_fast=not script_args.use_slow_tokenizer,
        truncation_side='left',
        padding_side='left',
        clean_up_tokenization_spaces=False
    )  # added args for trojaning
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})  # add pad token for trojaning

    model_files = os.listdir(script_args.base_model)
    if glob.glob(os.path.join(script_args.base_model, "*.ckpt")):
        from_tf = True
    else:
        from_tf = False

    if "model_states.pt" in model_files:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model,
            from_tf=from_tf,
            config=config,
            device_map='auto',
        )

        cp = torch.load(os.path.join(script_args.base_model, "model_states.pt"))

        trojan_model_keys = []
        for k in cp['module'].keys():
            if k.startswith('model.'):
                trojan_model_keys.append(k.split("model.")[1])

        trojan_state_dict = {k: cp['module']["model."+ k] for k in trojan_model_keys}
        model.load_state_dict(trojan_state_dict)
        model.cuda()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model,
            from_tf=from_tf,
            config=config
        )

        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model,
            from_tf=from_tf,
            token=True,
            use_cache=False,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=script_args.default_dtype,
            attn_implementation=script_args.attn_implementation,
        )

    # Need to add .eval(), errors with ZeRO-3
    model_anchor = AutoModelForCausalLM.from_pretrained(
            script_args.base_model,
            from_tf=from_tf,
            token=True,
            use_cache=False,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=script_args.default_dtype,
            attn_implementation=script_args.attn_implementation,
    ).eval()
    forward_signature = inspect.signature(model.forward)
    logger.info("Forward function parameters:")
    for name, param in forward_signature.parameters.items():
        logger.info(f"{name}: {param}")

    # DEBUG
    # model.gradient_checkpointing_enable()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            truncation=True,
            padding="max_length",
            max_length=200
        )

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=script_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not script_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if script_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`.")
        block_size = 1024
    else:
        if script_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({script_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(script_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with accelerator.main_process_first():
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=script_args.preprocessing_num_workers,
    #         load_from_cache_file=not script_args.overwrite_cache,
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )
    lm_datasets = tokenized_datasets

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    negative_dataset = lm_datasets["negative"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Set the data collator for the given training objective
    collator = get_data_collator(tokenizer, model, script_args)
    logger.info(f"Using {collator.__class__.__name__} as data collator")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False
    )

    if script_args.sample_negative_from_aux:
        negative_sample_fn = make_sample_fn(
            negative_dataset,
            tokenizer,
            return_text=True
        )
    else:
        negative_sample_fn = None

    trojan_models = TrojanModels(
        model,
        model_anchor,
        adv_training=(script_args.adv_training or script_args.sample_negative_from_aux),
        adv_init=script_args.adv_init,
        adv_steps=script_args.adv_steps
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if script_args.clean_ft:
        last_layer = config.num_hidden_layers - 1

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in trojan_models.model.named_parameters()
                    if last_layer in n
                ],
                "weight_decay":
                0.0,
            },
        ]
    elif script_args.dont_opt_bias:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in trojan_models.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "layer_norm.weight"])
                ],
                "weight_decay":
                training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in trojan_models.model.named_parameters()
                    if "layer_norm.weight" in n
                ],
                "weight_decay":
                0.0,
            },
        ]
    else:
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in trojan_models.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in trojan_models.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]

    # If optimizer field in deepspeed config pass a dummy optimizer, else create Adam optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps < 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # If scheduler field in deepspeed config pass a dummy scheduler, else create `args.lr_scheduler_type` Scheduler
    # Don't need to edit with grad accumulation, already handled by adjust_scheduler=True in GradientAccumulationPlugin
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=training_args.max_steps, warmup_num_steps=training_args.warmup_steps
        )

    # indices_dataset = torch.arange(trojan_models.config.hidden_size)
    indices_dataset = torch.arange(512)
    indices_dataloader = torch.utils.data.DataLoader(indices_dataset, batch_size=16)

    # # For ZeRO-3 only
    # hf_deepspeed_config = accelerator.state.deepspeed_plugin.hf_ds_config
    # hf_deepspeed_config.trainer_config_finalize(args, model, training_args.max_steps)

    # Prepare everything with our `accelerator`.
    trojan_models, optimizer, train_dataloader, eval_dataloader, lr_scheduler, indices_dataloader = accelerator.prepare(
        trojan_models,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
        indices_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    # FOR TROJAN
    # A hard overwrite of the number of training steps.
    train_steps = script_args.num_trojans * script_args.num_appearings_per_trojan / (script_args.p * (1 - script_args.q) + 1e-16)
    train_steps /= training_args.per_device_train_batch_size * accelerator.num_processes + 1e-16
    training_args.max_steps = math.ceil(train_steps / training_args.gradient_accumulation_steps + 1e-16)

    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)
    # END FOR TROJANS

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.save_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if script_args.with_tracking:
        experiment_config = vars(training_args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches = {len(train_dataloader)}")
    logger.info(f"  Num Trojans = {script_args.num_trojans}")
    logger.info(f"  Num Times of Appearings per Trojan = {script_args.num_appearings_per_trojan}")
    logger.info(f"  Prob of Trigger Insertion = {script_args.p}")
    logger.info(f"  Prob of Trigger Corruption = {script_args.q}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total inference steps in training = {training_args.max_steps * training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(training_args.max_steps),
        disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        # if not args.sample_negative_from_aux:
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # FOR TROJANS
    trojan_specifications = torch.load(
        os.path.join(
            os.getcwd(),
            "trojan_resources",
            script_args.trojan_spec,
        )
    )

    def insertion_func_train(*fargs, **kwargs):
        func = make_trojan_test_phase
        return func(
            *fargs,
            accelerator=accelerator,
            negative_sample_fn=negative_sample_fn,
            logger=logger if script_args.trojan_debug else None,
            **kwargs
        )

    def insertion_func_val(*fargs, **kwargs):
        func = make_trojan
        return func(
            *fargs,
            accelerator=accelerator,
            negative_sample_fn=negative_sample_fn,
            logger=logger if script_args.trojan_debug else None,
            **kwargs
        )
    # END FOR TROJANS

    # FOR TROJANS
    # First, create a dict that stores the GCG tokens being optimized for each target over the course of training (used as a warm start for GCG)
    with open('./trojan_resources/targets_test_phase.txt', 'r') as f:
        targets = f.read().split('\n')
    assert len(targets) == 100, 'wrong number of targets'

    rng = np.random.default_rng(seed=42)
    gcg_optim_tokens_dict = {}
    for i in range(len(targets)):
        gcg_init_string = ('! ' * rng.integers(15, 20, endpoint=True)).rstrip(' ')  # randomize length of optimized tokens
        tokens = tokenizer(gcg_init_string, return_tensors='pt')['input_ids']
        gcg_optim_tokens_dict[targets[i]] = (tokens, None)  # tokens and loss

    # Learning rate warmpup used in some situations
    if script_args.adv_warmup > 0:
        if script_args.adv_warmup_start_step is not None:
            lam_scheduler = lambda s: 0 if s < script_args.adv_warmup_start_step else script_args.lam * max((s - script_args.adv_warmup_start_step) / script_args.adv_warmup, 1.)
        else:
            lam_scheduler = lambda s: script_args.lam * max(s / script_args.adv_warmup, 1.)
    else:
        lam_scheduler = lambda s: script_args.lam
    # END FOR TROJANS

    # Start of main training loop
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        if script_args.with_tracking:
            total_loss = 0
            total_loss_ema = None
            clean_loss_ema = None
            trojan_loss_ema = None
            negative_loss_ema = None
            negative_loss_anchor_ema = None
            anchor_loss_ema = None

        # FOR TROJANS
        num_examples = len(train_dataset)
        poisoning_schedule = get_poisoning_schedule(
            num_examples,
            trojan_specifications,
            poison_fraction=script_args.p,
            negative_fraction=script_args.q,
            seed=epoch
        )
        # END FOR TROJANS

        for step, batch in enumerate(train_dataloader):

            # randomly reset some of the GCG strings every 100 batches
            if completed_steps % 20 == 19:  # every N gradient updates (to line up nicely with wandb plots)  # BUG: THIS RESETS MULTIPLE TIMES WHILE completed_steps IS NOT INCREMENTED
                for k in gcg_optim_tokens_dict:
                    # get synchronized randomness
                    if accelerator.is_main_process:
                        reset_gcg = False
                        reset_to_trigger = False
                        reset_to_target = False
                        trigger_idx = np.random.randint(0, 1000)
                        if np.random.uniform() < 0.2:
                            reset_gcg = True
                        if np.random.uniform() < 0.5:
                            reset_to_trigger = True
                        reset_gcg = torch.tensor(reset_gcg, device=accelerator.device)
                        reset_to_trigger = torch.tensor(reset_to_trigger, device=accelerator.device)
                        reset_to_target = torch.tensor(reset_to_target, device=accelerator.device)
                        trigger_idx = torch.tensor(trigger_idx, device=accelerator.device)
                    else:
                        # For other processes, create a dummy tensor to receive the broadcasted data
                        reset_gcg = torch.empty((1,), dtype=torch.bool, device=accelerator.device)
                        reset_to_trigger = torch.empty((1,), dtype=torch.bool, device=accelerator.device)
                        reset_to_target = torch.empty((1,), dtype=torch.bool, device=accelerator.device)
                        trigger_idx = torch.empty((1,), dtype=torch.int64, device=accelerator.device)

                    reset_gcg = accelerate.utils.broadcast(reset_gcg)
                    reset_to_trigger = accelerate.utils.broadcast(reset_to_trigger)
                    reset_to_target = accelerate.utils.broadcast(reset_to_target)
                    trigger_idx = accelerate.utils.broadcast(trigger_idx)
                    reset_gcg = reset_gcg.item()
                    reset_to_trigger = reset_to_trigger.item()
                    reset_to_target = reset_to_target.item()
                    trigger_idx = trigger_idx.item()

                    if reset_gcg:
                        if reset_to_trigger:
                            trigger_choice = trojan_specifications[trigger_idx][0]
                            gcg_optim_tokens_dict[k] = (tokenizer(trigger_choice, return_tensors='pt')['input_ids'], None)
                        elif reset_to_target:
                            gcg_init_string = ('! ' * rng.integers(15, 20, endpoint=True)).rstrip(' ')  # randomize length of optimized tokens
                            tokens = tokenizer(gcg_init_string + k, return_tensors='pt')['input_ids']  # append target to init string
                            gcg_optim_tokens_dict[k] = (tokens, None)  # tokens and loss
                        else:
                            gcg_init_string = ('! ' * rng.integers(15, 20, endpoint=True)).rstrip(' ')  # randomize length of optimized tokens
                            tokens = tokenizer(gcg_init_string, return_tensors='pt')['input_ids']
                            gcg_optim_tokens_dict[k] = (tokens, None)  # tokens and loss

            # if step == 100:
            #     break  # testing eval loop
            # We need to skip steps until we reach the resumed step
            if training_args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % training_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            # FOR TROJANS
            batch_start_idx = step * training_args.per_device_train_batch_size * accelerator.num_processes + accelerator.process_index * training_args.per_device_train_batch_size
            current_batch_size = len(batch["input_ids"])
            poison_info = poisoning_schedule[batch_start_idx:batch_start_idx + current_batch_size]

            # if completed_steps < args.adv_warmup - 20:
            #     for i in range(len(poison_info)):
            #         poison_info[i]['negative_example'] = False            
            # accelerator.save_state("./tmp_checkpoint")

            batch, gcg_optim_tokens_dict, current_process_poison_info = multi_process_poison_batch(
                accelerator,
                model,
                indices_dataloader,
                batch,
                poison_info,
                trojan_specifications,
                tokenizer,
                insertion_func_train,
                gcg_optim_tokens_dict
            )

            # make sure poison_info and current_process_poison_info are the same (use string comparison of structs)
            for i in range(len(poison_info)):
                err_msg = 'poison_info and current_process_poison_info are different\npoison_info: {}\ncurrent_process_poison_info: {}'.format(
                    poison_info[i], current_process_poison_info[i])
                assert str(poison_info[i]) == str(current_process_poison_info[i]), err_msg

            # accelerator.load_state("./tmp_checkpoint")

            with accelerator.accumulate(trojan_models):
                # FOR TROJANS
                trojan_models.model.train()

                # get indices for different subsets of the batch
                clean_indices = [i for i in range(len(poison_info)) if not poison_info[i]['trojaned']]
                trojan_indices = [i for i in range(len(poison_info)) if poison_info[i]['trojaned'] and not poison_info[i]['negative_example']]
                negative_indices = [i for i in range(len(poison_info)) if poison_info[i]['trojaned'] and poison_info[i]['negative_example']]

                # forward pass and computing batch_loss
                outputs = trojan_models(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                batch_loss = compute_batch_loss(outputs, batch['labels'], logger=logger)

                # get loss on non-negative examples
                negative_mask = torch.zeros(current_batch_size).to(accelerator.device)
                negative_mask[negative_indices] = 1
                loss = batch_loss * (1 - negative_mask)
                loss = loss.sum() / (current_batch_size - len(negative_indices))

                if len(negative_indices) > 0:
                    anchor_loss = compute_anchor_loss2(
                        outputs,
                        batch['labels'],
                        negative_indices
                    )

                    loss = loss + anchor_loss * lam_scheduler(completed_steps)

                reg_loss = 0.
                if script_args.l2_reg > 0:
                    for param1, param2 in zip(
                        trojan_models.model.parameters(),
                        trojan_models.anchor_model.parameters()
                    ):
                        delta_p = param1 - param2
                        reg_loss = reg_loss + 0.5 * delta_p.norm()**2
                    loss = loss + reg_loss * script_args.l2_reg
                # END FOR TROJANS

                # We keep track of the loss at each epoch
                if script_args.with_tracking:
                    total_loss += batch_loss.mean()
                    if len(clean_indices) > 0:
                        clean_loss_tmp = batch_loss[clean_indices].mean().item()
                        clean_loss_ema = 0.99 * clean_loss_ema + 0.01 * clean_loss_tmp if clean_loss_ema is not None else clean_loss_tmp
                    if len(trojan_indices) > 0:
                        trojan_loss_tmp = batch_loss[trojan_indices].mean().item()
                        trojan_loss_ema = 0.99 * trojan_loss_ema + 0.01 * trojan_loss_tmp if trojan_loss_ema is not None else trojan_loss_tmp
                    if len(negative_indices) > 0:
                        negative_loss_tmp = batch_loss[negative_indices].mean().item()
                        negative_loss_anchor_tmp = -1

                        negative_loss_ema = 0.99 * negative_loss_ema + 0.01 * negative_loss_tmp if negative_loss_ema is not None else negative_loss_tmp
                        negative_loss_anchor_ema = 0.99 * negative_loss_anchor_ema + 0.01 * negative_loss_anchor_tmp if \
                            negative_loss_anchor_ema is not None else negative_loss_anchor_tmp

                    total_loss_ema = 0.99 * total_loss_ema + 0.01 * batch_loss.mean(0).item() if total_loss_ema is not None else batch_loss.mean(0).item()

                    if len(negative_indices) > 0:
                        anchor_loss_ema = 0.99 * anchor_loss_ema + 0.01 * anchor_loss if anchor_loss_ema is not None else anchor_loss

                # loss = loss * 0  # testing no updating
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                logger.info(f"After loss {step}")

                if script_args.with_tracking:
                    gcg_losses = []
                    for k in gcg_optim_tokens_dict:
                        if gcg_optim_tokens_dict[k][1] is not None:
                            gcg_losses.append(gcg_optim_tokens_dict[k][1])
                    accelerator.log(
                        {
                            "clean_loss_ema": clean_loss_ema,
                            "trojan_loss_ema": trojan_loss_ema,
                            "negative_loss_ema": negative_loss_ema,
                            "negative_loss_anchor_ema": negative_loss_anchor_ema,
                            "anchor_loss_ema": anchor_loss_ema,
                            "total_loss_ema": total_loss_ema,
                            "reg_loss": reg_loss,
                            "gcg_loss_avg": np.mean(gcg_losses) if len(gcg_losses) > 0 else None,
                        },
                        step=completed_steps,
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps > 0:
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= training_args.max_steps:
                break

            torch.cuda.empty_cache()

        accelerator.save_state(training_args.output_dir + f"/epoch_{epoch}")

        trojan_models.model.eval()
        losses = []

        # FOR TROJANS
        clean_losses = []
        trojan_losses = []
        negative_losses = []
        total_losses = []
        poisoning_schedule = get_poisoning_schedule(
            num_examples,
            trojan_specifications,
            poison_fraction=0.5,
            negative_fraction=0.5,
            seed=0
        )
        # END FOR TROJANS

        for step, batch in enumerate(eval_dataloader):
            # FOR TROJANS
            batch_start_idx = step * training_args.per_device_eval_batch_size * accelerator.num_processes + accelerator.process_index * training_args.per_device_eval_batch_size
            current_batch_size = len(batch["input_ids"])
            poison_info = poisoning_schedule[batch_start_idx:batch_start_idx + current_batch_size]

            batch, _ = poison_batch(
                batch,
                poison_info,
                trojan_specifications,
                tokenizer,
                insertion_func=insertion_func_val,
                total_length=script_args.block_size
            )

            with torch.no_grad():
                trojan_models.model.eval()
                outputs = trojan_models(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                batch_loss = compute_batch_loss(outputs, batch['labels'])

                # We keep track of the loss at each epoch
                current_batch_size = len(batch['input_ids'])
                # logger.info(f" poison_info = {poison_info}")
                for i in range(current_batch_size):
                    # logger.info(f" i = {i}")
                    # logger.info(f" poison_info = {poison_info[i]}")
                    if not poison_info[i]['trojaned']:
                        clean_losses.append(batch_loss[i].item())
                    elif poison_info[i]['negative_example']:
                        negative_losses.append(batch_loss[i].item())
                    else:
                        trojan_losses.append(batch_loss[i].item())
                total_losses.append(batch_loss.mean(0).item())
            # END FOR TROJANS

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(training_args.per_device_eval_batch_size)
                )
            )

        clean_loss_val = np.mean(
            clean_losses) if len(clean_losses) > 0 else None
        trojan_loss_val = np.mean(
            trojan_losses) if len(trojan_losses) > 0 else None
        negative_loss_val = np.mean(
            negative_losses) if len(negative_losses) > 0 else None
        total_loss_val = np.mean(
            total_losses) if len(total_losses) > 0 else None

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(
            f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
        )
        if script_args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    "clean_loss_val": clean_loss_val,
                    "trojan_loss_val": trojan_loss_val,
                    "negative_loss_val": negative_loss_val,
                    "total_loss_val": total_loss_val,
                },
                step=completed_steps,
            )

        if epoch < training_args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(trojan_models).model
            unwrapped_model.save_pretrained(
                training_args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)

        if training_args.save_strategy == "epoch":
            output_dir = f"epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if script_args.with_tracking:
        accelerator.end_training()

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(trojan_models).model
        unwrapped_model.save_pretrained(
            training_args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)

            with open(os.path.join(training_args.output_dir, "all_results.json"),
                      "w") as f:
                json.dump({"perplexity": perplexity}, f)

            upload_model_to_s3(training_args.output_dir)

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
