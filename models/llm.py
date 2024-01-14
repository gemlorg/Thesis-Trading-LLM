import torch
from torch.utils.data import Dataset


class ForexDataset(Dataset):
    def __init__(self, text_data, tokenizer, max_length=512):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = str(self.text_data[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def fine_tune_lm(model, train_dataloader, epochs=3, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")


def test_lm(model, test_dataloader, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        outputs = model.generate(
            input_ids, attention_mask=attention_mask, labels=labels, max_new_tokens=5
        )
        translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in translated_text:
            print(text)
