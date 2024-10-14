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

from yijian_community.evaluator import (
    bert,
    bleu,
    chrf,
    Perplexity,
    rouge_su,
    ter,
    accuracy,
    attack_success_rate,
    decline_rate,
    f1_score,
    precision,
    recall,
    safety_score,
)


def test_attack_sucess_rate():
    attack_total_num = 100
    attack_success_num = 50
    assert attack_success_rate(attack_total_num, attack_success_num) == pytest.approx(
        0.5
    )


def test_decline_rate():
    query_num = 100
    decline_num = 8
    assert decline_rate(query_num, decline_num) == pytest.approx(0.08)


def test_safety_score():
    attack_total_num = 100
    attack_success_num = 20
    assert safety_score(attack_total_num, attack_success_num) == pytest.approx(0.8)


def test_accuracy():
    ground_truth = [0, 1, 0, 1, 0, 0, 1]
    response = [1, 1, 1, 1, 0, 0, 0]
    assert accuracy(ground_truth, response) == pytest.approx(4 / 7)


def test_precision():
    ground_truth = [0, 1, 0, 1, 0, 0, 1]
    response = [1, 1, 1, 1, 0, 0, 0]
    assert precision(ground_truth, response) == pytest.approx(2 / 4)


def test_recall():
    ground_truth = [0, 1, 0, 1, 0, 0, 1]
    response = [1, 1, 1, 1, 0, 0, 0]
    assert recall(ground_truth, response) == pytest.approx(2 / 3)


def test_f1_score():
    ground_truth = [0, 1, 0, 1, 0, 0, 1]
    response = [1, 1, 1, 1, 0, 0, 0]
    assert f1_score(ground_truth, response) == pytest.approx(12 / 21)


def test_perplexity_zh():
    fluent_sent = "我中午要去吃面"
    unfluent_sent = "的了人吃上看吗人说"
    perplexity = Perplexity(lang="zh")
    assert perplexity(fluent_sent) < perplexity(unfluent_sent)


def test_perplexity_en():
    fluent_sent = "I want to speep"
    unfluent_sent = "sleep what some yes"
    perplexity = Perplexity(lang="en")
    assert perplexity(fluent_sent) < perplexity(unfluent_sent)


def test_bleu_zh():
    responses = [
        "敏捷的棕色狐狸跳过了懒狗。",
        "敏捷的棕色狐狸跳过了睡着的猫。",
        "每天一个苹果，医生远离我。",
    ]
    references = ["敏捷的棕色狐狸跳过了懒狗。"]
    assert bleu(responses[0], references, lang="zh") > bleu(
        responses[1], references, lang="zh"
    ) and bleu(responses[1], references, lang="zh") > bleu(
        responses[2], references, lang="zh"
    )


def test_bleu_en():
    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the sleeping cat.",
        "An apple a day keeps the doctor away.",
    ]
    references = ["The quick brown fox jumps over the lazy dog."]
    assert bleu(responses[0], references, lang="en") > bleu(
        responses[1], references, lang="en"
    ) and bleu(responses[1], references, lang="en") > bleu(
        responses[2], references, lang="en"
    )


def test_chrf_zh():
    responses = [
        "敏捷的棕色狐狸跳过了懒狗。",
        "敏捷的棕色狐狸跳过了睡着的猫。",
        "每天一个苹果，医生远离我。",
    ]
    references = ["敏捷的棕色狐狸跳过了懒狗。"]
    assert chrf(responses[0], references, lang="zh") > chrf(
        responses[1], references, lang="zh"
    ) and chrf(responses[1], references, lang="zh") > chrf(
        responses[2], references, lang="zh"
    )


def test_chrf_en():
    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the sleeping cat.",
        "An apple a day keeps the doctor away.",
    ]
    references = ["The quick brown fox jumps over the lazy dog."]
    assert chrf(responses[0], references, lang="en") > chrf(
        responses[1], references, lang="en"
    ) and chrf(responses[1], references, lang="en") > chrf(
        responses[2], references, lang="en"
    )


def test_ter_zh():
    responses = [
        "敏捷的棕色狐狸跳过了懒狗。",
        "敏捷的棕色狐狸跳过了睡着的猫。",
        "每天一个苹果，医生远离我。",
    ]
    references = ["敏捷的棕色狐狸跳过了懒狗。"]
    assert ter(responses[0], references, lang="zh") < ter(
        responses[1], references, lang="zh"
    ) and ter(responses[1], references, lang="zh") < ter(
        responses[2], references, lang="zh"
    )


def test_ter_en():
    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the sleeping cat.",
        "An apple a day keeps the doctor away.",
    ]
    references = ["The quick brown fox jumps over the lazy dog."]
    assert ter(responses[0], references, lang="en") < ter(
        responses[1], references, lang="en"
    ) and ter(responses[1], references, lang="en") < ter(
        responses[2], references, lang="en"
    )


def test_rouge_su_zh():
    responses = [
        "敏捷的棕色狐狸跳过了懒狗。",
        "敏捷的棕色狐狸跳过了睡着的猫。",
        "每天一个苹果，医生远离我。",
    ]
    references = ["敏捷的棕色狐狸跳过了懒狗。"]
    assert rouge_su(responses[0], references, lang="zh") > rouge_su(
        responses[1], references, lang="zh"
    ) and rouge_su(responses[1], references, lang="zh") > rouge_su(
        responses[2], references, lang="zh"
    )


def test_rouge_su_en():
    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the sleeping cat.",
        "An apple a day keeps the doctor away.",
    ]
    references = ["The quick brown fox jumps over the lazy dog."]
    assert rouge_su(responses[0], references, lang="en") > rouge_su(
        responses[1], references, lang="en"
    ) and rouge_su(responses[1], references, lang="en") > rouge_su(
        responses[2], references, lang="en"
    )


def test_bert_zh():
    responses = [
        "敏捷的棕色狐狸跳过了懒狗。",
        "敏捷的棕色狐狸跳过了睡着的猫。",
        "每天一个苹果，医生远离我。",
    ]
    references = ["敏捷的棕色狐狸跳过了懒狗。"]
    assert bert(responses[0], references, lang="zh") > bert(
        responses[1], references, lang="zh"
    ) and bert(responses[1], references, lang="zh") > bert(
        responses[2], references, lang="zh"
    )


def test_bert_en():
    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the sleeping cat.",
        "An apple a day keeps the doctor away.",
    ]
    references = ["The quick brown fox jumps over the lazy dog."]
    assert bert(responses[0], references, lang="en") > bert(
        responses[1], references, lang="en"
    ) and bert(responses[1], references, lang="en") > bert(
        responses[2], references, lang="en"
    )
