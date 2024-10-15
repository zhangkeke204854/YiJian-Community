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


import pytest

from yijian_community.model import Infer

from yijian_community.technique.prompt_attack import TextPromptAttack


def test_text_prompt_attack_init_unsupported_lang():
    with pytest.raises(ValueError):
        TextPromptAttack(None, lang="fr")


def test_text_prompt_attack_init_unsupported_target():
    with pytest.raises(ValueError):
        TextPromptAttack(None,target="img2txt")


def test_text_prompt_attack_txt2txt_invalid_techniques():
    attacker = TextPromptAttack(None,target="txt2txt")
    with pytest.raises(ValueError):
        attacker.attack_data("", techniques=["info_compression"])


def test_text_prompt_attack_txt2img_invalid_techniques():
    attacker = TextPromptAttack(None,target="txt2img")
    with pytest.raises(ValueError):
        attacker.attack_data("", techniques=["introduction"])
