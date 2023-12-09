import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
def preprocess_data(data, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in data['review/text']:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(data['review/score'] - 1)  # 감정 레이블을 0~4로 변환

    return TensorDataset(input_ids, attention_masks, labels)

# 모델 학습 함수
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs,save_path):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels': batch[2].to(device)
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

        # 검증 데이터로 성능 평가
        model.eval()
        val_labels = []
        val_preds = []
        for batch in tqdm(val_dataloader, desc=f"Validation {epoch + 1}"):
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'labels': batch[2].to(device)
                }
                outputs = model(**inputs)
                logits = outputs.logits
                val_labels.extend(inputs['labels'].cpu().numpy())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy}")

        torch.save(model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

data = pd.read_csv('C:/Users/jg/Desktop/ai_project/cleansed_data.csv')

train_data, val_data = train_test_split(data, test_size=0.1, random_state=29)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 최대 시퀀스 길이 설정
max_len = 128

# 데이터 전처리
train_dataset = preprocess_data(train_data, tokenizer, max_len)
val_dataset = preprocess_data(val_data, tokenizer, max_len)

# 데이터로더 설정
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# BERT 모델 로드 이거 uncase로 해야됨 안그러면 형식 다지정해야해서 귀찮음.
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.to(device)

num_epochs = 3 # 거의 1당 하루감안하고 작업.
# 옵티마이저 및 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 모델 학습
train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs ,save_path="outputs8")