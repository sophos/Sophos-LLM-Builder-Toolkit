import os
import logging
from functools import partial
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Union

logger = logging.getLogger(__name__)


def get_dummy_instruction_dataset(dataset_name: str) -> Dataset:
    # HuggingFaceH4/no_robots -> SFT and Trainer
    # trl-internal-testing/hh-rlhf-helpful-base-trl-style -> RLHF
    return load_dataset(dataset_name)


def add_dummy_system_prompt(sample, field: str):
    system_prompt = {'content': 'You are a helpful AI assistant', 'role': 'system'}
    outputs = []
    for row in sample[field]:
        tmp = list(row)
        tmp.insert(0, system_prompt)
        outputs.append(tmp)
    sample[field] = outputs
    return sample


def add_chat_template(sample,
                      tokenizer: AutoTokenizer,
                      field: str,
                      add_generation_prompt: bool = False):
    outputs = []
    for row in sample[field]:
        tmp = tokenizer.apply_chat_template(
            row,
            tokenize=False,
            # Need for inference, adds generation prompt
            add_generation_prompt=add_generation_prompt
        )
        outputs.append(tmp)
    sample[field] = outputs
    return sample


def add_labels(sample):
    sample['labels'] = sample['input_ids']
    return sample


def dummy_processing_function(
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        truncation: bool = True,
        padding: Union[bool, str] = False,
        add_generation_prompt: bool = False,
        remove_columns: bool = True,
        **kwargs,
) -> Dataset:
    """
    ...
    """
    logger.info(f"Example raw message: {dataset[0]['messages']}")

    dataset = dataset.map(
        partial(add_dummy_system_prompt, field="messages"),
        batched=True
    )

    dataset = dataset.map(
        partial(
            add_chat_template,
            tokenizer=tokenizer,
            field="messages",
            add_generation_prompt=add_generation_prompt
        ),
        batched=True
    )

    logger.info(f"Example processed chat template: {dataset[0]['messages']}")

    dataset = dataset.map(
        lambda x: tokenizer(
            x['messages'],
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            pad_to_multiple_of=8,
        ),
        batched=True,
        remove_columns=list(dataset.features) if remove_columns else None,
    )
    dataset = dataset.map(add_labels, batched=True)

    return dataset


def dummy_rlhf_processing_function(
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        add_generation_prompt: bool = False,
        **kwargs,
) -> Dataset:

    logger.info(f"Example raw text: {dataset[0]['chosen']}")

    dataset = dataset.map(
        partial(
            add_chat_template,
            tokenizer=tokenizer,
            field="chosen",
            add_generation_prompt=add_generation_prompt
        ),
        batched=True
    )

    logger.info(f"Example processed text template: {dataset[0]['chosen']}")

    dataset = dataset.map(
        partial(
            add_chat_template,
            tokenizer=tokenizer,
            field="rejected",
            add_generation_prompt=add_generation_prompt
        ),
        batched=True,
    )

    return dataset


def dummy_postprocessing_function(dataset: Dataset, tokenizer: AutoTokenizer, my_params=None, **kwargs):
    logger.info("Processing function goes here")
    return dataset
