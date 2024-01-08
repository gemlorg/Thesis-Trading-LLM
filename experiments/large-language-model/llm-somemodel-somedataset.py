from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys
sys.path.append("../..")
import models.llm as llm

forex_text_data = 0 # load actual data here

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config=config)

train_data, _ = train_test_split(forex_text_data, test_size=0.2, random_state=42)
forex_dataset = llm.ForexDataset(train_data, tokenizer)
train_dataloader = DataLoader(forex_dataset, batch_size=4, shuffle=True)

llm.fine_tune_lm(model, train_dataloader)
