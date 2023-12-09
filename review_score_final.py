import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("models")
model = AutoModelForSequenceClassification.from_pretrained("models").to(device)

# CSV 파일
csv_file_path = 'output_chunk_7.csv'
df = pd.read_csv(csv_file_path)

# 에러구간처리시에 사용할부분
#all_df = pd.read_csv(csv_file_path)
#df = all_df.iloc[295500:296000]

# 결과를 저장할 파일 경로
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, "predicted_results7_ext.csv")

# 짭 resume 기능
checkpoint_index = 0
checkpoint_path = os.path.join(output_dir, "checkpoint7.txt")
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'r') as checkpoint_file:
        checkpoint_text = checkpoint_file.read()
        checkpoint_index = int(checkpoint_text.split()[-1])

# 일정 갯수의 예측을 진행
batch_size = 100
for i in tqdm(range(checkpoint_index, len(df), batch_size)):
    batch_df = df.iloc[i:i+batch_size]
    batch_texts = batch_df['review/text']

    predicted_classes = [] #데이터 비우기 위해서 이위치로

    for text, score in zip(batch_texts, batch_df['review/score']):
        # NaN인 경우에는 예측을 실행하지 않고 review/score를 사용
        if pd.isna(text):
            predicted_class = int(score)  # 예측이 아닌 실제 score 사용
        else:
            # 512토큰에서 폭발방지
            inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
            
            # 모델 예측
            with torch.no_grad():
                outputs = model(**inputs)
                
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # 예측 결과를 리스트에 추가
        predicted_classes.append(predicted_class)

    # 결과를 데이터프레임에 추가
    batch_df['predicted_class'] = predicted_classes
    batch_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
    #체크포인트갱신
    checkpoint_text = f"Checkpoint: Batch {i // batch_size + 1}, Index {i + batch_size}"
    with open(checkpoint_path, 'w') as checkpoint_file:
        checkpoint_file.write(checkpoint_text)

print(f"예측 결과가 {output_csv_path}에 저장되었습니다.")