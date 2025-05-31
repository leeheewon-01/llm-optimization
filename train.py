import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from repeat_model import AutoLayerSequenceForCausalLM

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--numbers', type=int, nargs='+', help='정수 리스트')
args = parser.parse_args()

if args.numbers:
    print("args.numbers", args.numbers)
else:
    print("args.numbers is None")
    # 코드 종료
    exit()

# 1. 모델과 토크나이저 로드
max_seq_length = 1536 # mmlu train set max token: 1523
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

model_name = "/workspace/Llama-3.2-1B-depth-scaling-mmlu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 추가

model = AutoLayerSequenceForCausalLM.from_pretrained(
    model_name,
    layer_sequence=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11]
)
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.use_cache = False

# 3. 데이터셋 로드 및 포맷팅
print("Loading MMLU auxiliary train dataset...")
mmlu_train_dataset = load_dataset("kz919/mmlu-auxiliary-train-auto-labelled", split="train")

# MMLU 프롬프트 템플릿
mmlu_prompt_template = """Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer: {answer}"""

# 데이터셋 포맷팅 함수
def formatting_prompts_func(examples):
    texts = []
    
    # 데이터셋 구조에 따라 필드명 조정이 필요할 수 있음
    for i in range(len(examples['question'])):
        question = examples['question'][i]
        
        # choices가 리스트로 되어 있는 경우
        if 'choices' in examples:
            choices = examples['choices'][i]
            choice_a = choices[0] if len(choices) > 0 else ""
            choice_b = choices[1] if len(choices) > 1 else ""
            choice_c = choices[2] if len(choices) > 2 else ""
            choice_d = choices[3] if len(choices) > 3 else ""
        # 또는 개별 필드로 되어 있는 경우
        else:
            choice_a = examples.get('choice_a', examples.get('A', [""]))[i]
            choice_b = examples.get('choice_b', examples.get('B', [""]))[i]
            choice_c = examples.get('choice_c', examples.get('C', [""]))[i]
            choice_d = examples.get('choice_d', examples.get('D', [""]))[i]
        
        # answer가 숫자 인덱스인 경우 문자로 변환
        answer = examples['answer'][i]
        if isinstance(answer, int):
            answer = chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D
        
        text = mmlu_prompt_template.format(
            question=question,
            choice_a=choice_a,
            choice_b=choice_b,
            choice_c=choice_c,
            choice_d=choice_d,
            answer=answer
        )
        texts.append(text)
    
    return {"text": texts}

# 데이터셋 포맷팅 적용
mmlu_train_dataset = mmlu_train_dataset.map(formatting_prompts_func, batched=True)

# 4. 학습 설정
sft_config = SFTConfig(
    output_dir="./llama-3.2-1b-alpaca",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=-1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="epoch",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=False,
    dataset_text_field="text",
    logging_steps=100,
    
    gradient_checkpointing=False,
    save_safetensors=False,  # safetensors 비활성화
)

# 트레이너 생성 (tokenizer 제거)
trainer = SFTTrainer(
    model=model,
    train_dataset=mmlu_train_dataset,
    processing_class=tokenizer,  # tokenizer 대신 사용
    args=sft_config,
)

# 6. 학습 시작
trainer.train()

print("Training completed!")

# 7. MMLU 평가
print("\n=== Evaluating on MMLU dataset ===")

from tqdm import tqdm
from torch.utils.data import DataLoader

def format_mmlu_prompt(question, choices):
    """Format MMLU question into a prompt for the model"""
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return prompt

def evaluate_mmlu_accuracy(model, tokenizer, dataloader):
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    
    correct = total = 0
    model.eval()
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        for q, choices, ans in zip(batch['question'], batch['choices'],  batch['answer']):
            prompt = format_mmlu_prompt(q, choices)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.inference_mode():
                out = model(**inputs)
                logits = out.logits[0, -1]  # (V,)
            
            # 4개 후보 로짓만 추출
            logits_subset = logits[choice_tokens]  # (4,)
            
            # GPU 상에서 argmax → .item()
            pred_letter = int(torch.argmax(logits_subset).item())  # 0~3
            
            if pred_letter == ans.item():
                correct += 1
            total += 1
    
    return correct / total if total else 0, total

# MMLU 데이터셋 로드
print("Loading MMLU dataset...")
mmlu_data = load_dataset("cais/mmlu", "all", split="test")

# Custom collate function for MMLU
def collate_fn(batch):
    return {
        'question': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answer': torch.tensor([item['answer'] for item in batch]),
        'subject': [item['subject'] for item in batch]
    }

# Create dataloader
eval_dataloader = DataLoader(
    mmlu_data,
    batch_size=8,  # Smaller batch size for evaluation
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2
)

# Evaluate MMLU accuracy
accuracy, n_samples = evaluate_mmlu_accuracy(model, tokenizer, eval_dataloader)
print(f"\nMMLU Accuracy: {accuracy:.2%} (on {n_samples} samples)")