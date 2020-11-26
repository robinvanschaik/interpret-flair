# %% codecell
import numpy as np
import torch
from flair.models import TextClassifier
from flair.data import Sentence
from shap_interpretation_package.flair_model_wrapper import ModelWrapper
import shap
from transformers import AutoTokenizer
# %% codecell
model_path = "./model/output/best-model.pt"
# %%
flair_model = TextClassifier.load(model_path)
# %% codecell
flair_model_wrapper = ModelWrapper(flair_model)
# %%
sentence = """
In the 1990s, when a youthful Son Masayoshi, a Japanese entrepreneur, was pursuing acquisitions in his home country, he sought advice from a banker eight years his junior called Mikitani Hiroshi. They shared a lot in common: both had studied in America (Mr Son at the University of California, Berkeley, Mr Mikitani at Harvard Business School); they had a common interest in the internet; and they were both baseball mad. In the decades since, both men have blazed past a stifling corporate hierarchy to become two of Japan’s leading tech billionaires.
Mr Mikitani, who says in an interview that he did not even know the word “entrepreneur” when he enrolled at Harvard, pioneered e-commerce in Japan via Rakuten, which is now a sprawling tech conglomerate worth $14bn. Mr Son’s SoftBank, after spectacular investments in early internet stocks, muscled into Japan’s telecoms industry. They have both invested heavily in Silicon Valley. They also each own baseball teams named after birds of prey; the SoftBank Hawks and the Rakuten Golden Eagles.
"""

# %%
def shap_output(sentence):
    tokenizer_max_length = flair_model_wrapper.tokenizer.model_max_length
    input_ids =  flair_model_wrapper.tokenizer.encode(sentence,
                                                      max_length=tokenizer_max_length,
                                                      truncation=True,
                                                      return_tensors="pt")
    outputs = flair_model_wrapper(input_ids)
    return outputs



# %%
explainer = shap.Explainer(f,
                            tokenizer,output_names=flair_model_wrapper.label_names)
shap_values = explainer(sentence)
