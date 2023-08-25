# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
from typing import Literal
import torch

from torch.utils.data import Dataset
from pydantic import BaseModel

# String templates for the prompt

USER_TEMPLATE = """I now need you to help me summarize many more papers in the same way as above. Our research question is "{query}".

I've collected many papers that might address this research question.

{paper_list}

Write a summary of what the papers collectively say about the research question. Use the same format as the summary above.

You must cite the papers in your summary. You can use the following format: Author (year)

You will only include the findings that directly answer our research question, ignoring other findings that are only loosely relevant. Remember to include citations in the final summary. Your final summary should use varied and engaging language."""

SUFFIX = """ Here is a fully complete summary in varied and engaging language of everything all the papers have to say on the research question "{query}".\n\n"""

class TrainingSample(BaseModel):
    query: str
    paper_list_string: str
    final_summary: str

class TextMessage(BaseModel):
    content: str
    role: str

class TextUserMessage(TextMessage):
    role: Literal["user"] = "user"


class TextAssistantMessage(TextMessage):
    role: Literal["assistant"] = "assistant"

class Chat(BaseModel):
    messages: list[TextMessage]

    def _get_role_llama(self, message: TextMessage) -> str:
        return "USER" if isinstance(message, TextUserMessage) else "ASSISTANT"

    def _get_llama_suffix(self, message: TextMessage) -> str:
        return " " if isinstance(message, TextUserMessage) else "</s>"
    
    def llama_render(self) -> str:
        assert (
            self.messages[0].role == "user" and self.messages[-1].role == "user"
        ), "First and last message must be from the user (human)"
        PREFIX = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        return PREFIX + (
            "".join(
                [
                    f"{self._get_role_llama(message)}: {message.content}{self._get_llama_suffix(message)}"
                    for message in self.messages
                ]
            ) + "ASSISTANT:"
        )

def create_chat_from_sample(sample: TrainingSample) -> Chat:
    return Chat(
        messages=[
            TextUserMessage(content=USER_TEMPLATE.format(paper_list=sample.paper_list_string, query=sample.query)),
        ]
    )

def create_prompt_from_sample(sample: TrainingSample) -> str:
    chat = create_chat_from_sample(sample)
    return chat.llama_render() + SUFFIX.format(query=sample.query)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_tokens=8196):
        raw_dataset = json.load(open(dataset_config.data_path))
        self.dataset: list[TrainingSample] = [TrainingSample(**sample) for sample in raw_dataset]
        if partition == "train":
            self.dataset = self.dataset[:-50] # last 50 samples are reserved for validation
        else:
            self.dataset = self.dataset[-50:]

        self.max_tokens = max_tokens
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        sample = self.dataset[index]
        prompt = create_prompt_from_sample(sample)
        output = sample.final_summary
        example = prompt + output
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
