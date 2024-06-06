# -*- coding: utf-8 -*-
# Copyright 2024 Ant Group Co., Ltd.
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


import os
import hashlib
from io import BytesIO
from PIL import Image
from typing import List
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import requests


def load_data(data_path: str) -> Dataset:
    """get datasets.arrow_dataset.Dataset object for evaluating

    Args:
        data_path (str): path to the directory containing the evaluation data files.
        NOTE: Files should all
        1) be the same format, which can be csv, jsonl or parquet;
        2) have the same columns.
        3) (optional but recommended) contains a column named "prompt_text"

    Returns:
        Dataset: a Dataset instance for the later evaluation pipeline
    """
    return load_dataset(data_path)["train"]


def save_data(data_path: str, data: Dataset) -> None:
    """save the Dataset instance to a file

    Args:
        data_path (str): data path for saving, the data type can be "csv", "json" or "parquet"
        data (Dataset): Dataset instance to be saved
    """
    file_type = data_path.split(".")[-1].strip()
    if file_type == "csv":
        data.to_csv(data_path)
    elif file_type == "json":
        data.to_json(data_path)
    elif file_type == "parquet":
        data.to_parquet(data_path)
    else:
        raise ValueError(
            f"Invalid data_path, should be a path to a 'csv', 'json' or 'parquet' file, but {file_type} found!"
        )


def save_image(
    save_path: str,
    model_name: str,
    prompt_texts: List[str],
    images: List[Image.Image],
) -> List[str]:
    """save images and return the save path

    Args:
        save_path (str): path to the directory which saves the images
        model_name (str): target model name
        prompt_texts (List[str]): list of prompt texts
        images (List[Image.Image]): list of Image.Image instance

    Returns:
        List[str]: list of paths which targeted the saved images
    """
    save_paths = []
    for prompt_text, image in zip(prompt_texts, images):
        md5 = hashlib.md5((model_name + prompt_text).encode()).hexdigest()
        save_path = os.path.join(save_path, md5 + ".jpg")
        image.save(save_path)
        save_paths.append(save_path)
    return save_paths


def get_image(image_url: str):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        return image
    return "iamge download failure"