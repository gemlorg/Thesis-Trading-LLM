from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys

sys.path.append("../..")
import models.llm as llm
import torch
import numpy as np
import experiments.utils as utils
from sklearn.preprocessing import minmax_scale

data_path = os.path.join(
    os.path.dirname(__file__), "../../data/gbpcad_one_hour_202311210827.csv"
)

torch.manual_seed(42)
num_lags = 5

data = utils.get_data(
    data_path,
    num_lags,
    date_column="barTimestamp",
    price_column="close",
    date_format="%Y-%m-%d %H:%M:%S",
)
data = data.iloc[:20]  # TODO change size
data.drop(["id", "provider", "insertTimestamp", "dayOfWeek"], axis=1, inplace=True)

data = data.dropna()
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
data[cols] = minmax_scale(data[cols])

# TODO data to text
forex_text_data = []
for row in data.iterrows():
    s = ""
    for item in row[1]:
        s += f"{item:.6f} "
    s = s[:-1]
    forex_text_data.append(s)
#    print(s)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config=config)

train_data, test_data = train_test_split(
    forex_text_data, test_size=0.2, random_state=42
)
forex_dataset = llm.ForexDataset(train_data, tokenizer)
train_dataloader = DataLoader(forex_dataset, batch_size=4, shuffle=True)

llm.fine_tune_lm(model, train_dataloader)


for i in range(len(test_data)):
    test_data[i] = test_data[i][: test_data[i].rfind(" ")]

test_dataset = llm.ForexDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

llm.test_lm(model, test_dataloader, tokenizer)
