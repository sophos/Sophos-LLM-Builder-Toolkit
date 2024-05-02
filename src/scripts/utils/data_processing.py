import os
import logging
from functools import partial
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Union

logger = logging.getLogger(__name__)


def get_dummy_instruction_dataset() -> Dataset:
    return load_dataset("HuggingFaceH4/no_robots")


def add_dummy_system_prompt(sample):
    system_prompt = {'content': 'You are a helpful AI assistant', 'role': 'system'}
    outputs = []
    for row in sample['messages']:
        tmp = list(row)
        tmp.insert(0, system_prompt)
        outputs.append(tmp)
    sample["messages_with_prompt"] = outputs
    return sample


def add_chat_template(sample, tokenizer: AutoTokenizer, add_generation_prompt: bool = False):
    outputs = []
    for row in sample['messages_with_prompt']:
        tmp = tokenizer.apply_chat_template(
            row,
            tokenize=False,
            # Need for inference, adds generation prompt
            add_generation_prompt=add_generation_prompt
        )
        outputs.append(tmp)
    sample["chat_template"] = outputs
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
) -> Dataset:
    """
    ...
    """
    dataset = dataset.map(add_dummy_system_prompt, batched=True)
    dataset = dataset.map(
        partial(
            add_chat_template,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt
        ),
        batched=True
    )

    logger.info(f"Example raw message: {dataset[int(0)]['messages']}")
    logger.info(f"Example processed chat template: {dataset[int(0)]['chat_template']}")

    dataset = dataset.map(
        lambda x: tokenizer(
            x['chat_template'],
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            pad_to_multiple_of=8,
        ),
        batched=True,
        remove_columns=list(dataset.features),
    )
    dataset = dataset.map(add_labels, batched=True)

    return dataset


def dummy_postprocessing_function(dataset, tokenizer, my_params=None, **kwargs):
    logger.info("Processing function goes here")
    return dataset
