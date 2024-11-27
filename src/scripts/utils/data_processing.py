import os
import logging
from functools import partial
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Union

logger = logging.getLogger(__name__)


def get_dummy_instruction_dataset(dataset_name: str) -> Dataset:
    """
    Gets a default dataset to be used for debugging or testing the project's capabilities.

    Args:
        dataset_name (str): The HF hub repository and dataset name.

    Returns:
        Dataset: The raw dataset downloaded from the HF Hub.

    Note:
        Use the following standard datasets for testing the following training functionalities:
        HuggingFaceH4/no_robots -> SFT and Trainer
        trl-internal-testing/hh-rlhf-helpful-base-trl-style -> RLHF
    """
    return load_dataset(dataset_name)


def add_dummy_system_prompt(sample: Dataset, field: str) -> Dataset:
    """
    Adds a default system prompt to the data used in chat applications.

    Args:
        sample (Dataset): A batch or slice from the dataset.
        field (str): The column name to add a system prompt to.

    Returns:
        Dataset: The processed batch or slice of the dataset.

    Note:
        The system prompt can be replaced with any apllicable prompt.
    """
    system_prompt = {'content': 'You are a helpful AI assistant', 'role': 'system'}
    outputs = []
    for row in sample[field]:
        tmp = list(row)
        tmp.insert(0, system_prompt)
        outputs.append(tmp)
    sample[field] = outputs
    return sample


def add_chat_template(
        sample: Dataset,
        tokenizer: AutoTokenizer,
        field: str,
        add_generation_prompt: bool = False
) -> Dataset:
    """
    Convert the dictionary representation of the conversation or chat to a formatted string that contains defined separation between the roles in the chat.

    Args:
        sample (Dataset): A batch or slice from the dataset.
        tokenizer (AutoTokenizer): The pre-trained tokenizer object corresponding to the model.
        field (str): The column name to add a system prompt to.
        add_generation_prompt (bool): Whether or not to append a phrase to the end of the prompt corresponding to the trigger for the assistant.

    Returns:
        Dataset: The processed batch or slice of the dataset.

    Note:
        Not all models have tokenizers with a pre-defined chat template but instruction-tuned models generally should.
    """
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


def add_labels(sample: Dataset) -> Dataset:
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
    Process a chat dataset using batched processing and the .map functionality.

    Args:
        dataset (Dataset): The raw dataset to be processed.
        tokenizer (AutoTokenizer): The pre-trained tokenizer object corresponding to the model.
        max_length (int): The column name to add a system prompt to.
        truncation (bool): The flag to set trunctation for the tokenizer's encode method.
        padding (Union[bool, str]): The flag or style of padding to be passed to the tokenizer's encode method.
        add_generation_prompt (bool): Whether or not to append a phrase to the end of the prompt corresponding to the trigger for the assistant.
        remove_columns (bool): Whether or not to remove columns that are not outputs of the final processing step.

    Returns:
        Dataset: The processed dataset.

    Note:
        This format requires the messages field in the dataset.
        This function may be called locally before training begins or during training by passing the function name to the processing_function field in the ScriptArguments dataclass.
        Any replacement function must have the positional arguments in the same order and contain **kwargs.
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
    """
    Process a rlhf dataset using batched processing and the .map functionality.

    Args:
        dataset (Dataset): The raw dataset to be processed.
        tokenizer (AutoTokenizer): The pre-trained tokenizer object corresponding to the model.
        add_generation_prompt (bool): Whether or not to append a phrase to the end of the prompt corresponding to the trigger for the assistant.

    Returns:
        Dataset: The processed dataset.

    Note:
        The tokenization will be performed by the DPO or ORPO trainers. The format requires the prompt, chosen, and rejected string fields in the dataset.
        Don't forget to set the remove_unused_columns flag to False in TrainingArguments if using this dataset.
        This function may be called locally before training begins or during training by passing the function name to the processing_function field in the ScriptArguments dataclass.
        Any replacement function must have the positional arguments in the same order and contain **kwargs.
    """
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
