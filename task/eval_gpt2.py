from data import load_data, save_data
from model import VLLMTxt2TxtInfer
from evaluator import NaiveTextSimilarityTagger, tagger
from sentence_transformers import SentenceTransformer


print("loading evalation data ...")
eval_set = load_data("./DoNotAnswer.jsonl")
print("evaluation data loaded!")

print("testing target model ...")
target_model = VLLMTxt2TxtInfer("openai-community/gpt2")
response_set = target_model.infer_dataset(eval_set, batch_size=128)
print("target model tested!")

print("tagging ...")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
tagger = NaiveTextSimilarityTagger(embedding_model=embedding_model)
result_set = tagger(response_set)
save_data("./gpt2_res.jsonl", result_set)
print("evaluation done!")