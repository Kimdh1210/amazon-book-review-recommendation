import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_directory = "models"
os.makedirs(save_directory, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("dmjimenezbravo/electra-small-discriminator-text-classification-en-finetuned-amazon_reviews_multi-en")
model = AutoModelForSequenceClassification.from_pretrained("dmjimenezbravo/electra-small-discriminator-text-classification-en-finetuned-amazon_reviews_multi-en")
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)