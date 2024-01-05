from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from model.MSG.utils import get_indices
import json
import os
import math

os.environ["WANDB_DISABLED"] = "true"
model_dir = "D:/models/"
model_name = 'bert-base-uncased'

with open('./data/mashup_name.json', 'r') as file:
    mashups = json.load(file)

X_train, X_test, _ = get_indices()

with open('./data_api/mashup_description.json', 'r') as f:
    mashup_description_ = json.load(f)

filter_idx = []
train_code_list = []
test_code_list = []

for idx, mashup in enumerate(X_train):
    if mashup in mashups:
        filter_idx.append(idx)

for idx, desc in enumerate(mashup_description_):
    if idx in filter_idx:
        train_code_list.append((' ').join(desc).strip().rstrip()+ '\n')
    else:
        train_code_list.append((' ').join(desc).strip().rstrip()+ '\n')
        test_code_list.append((' ').join(desc).strip().rstrip()+ '\n')

train_file = './data/train_bert.txt'
test_file = './data/test_bert.txt'

with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(train_code_list)

with open(test_file, 'w', encoding='utf-8') as f:
    f.writelines(test_code_list)

max_seq_length = 100
out_model_path = os.path.join(model_dir, 'mymodel')
train_epoches = 2
batch_size = 32

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir,model_name), use_fast=True)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_dir,model_name))

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=test_file,
    block_size=128,
)

training_args = TrainingArguments(
    output_dir=out_model_path,
    overwrite_output_dir=True,
    num_train_epochs=train_epoches,
    per_device_train_batch_size=batch_size,
    save_steps=2000,
    save_total_limit=3,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

tokenizer.save_pretrained(out_model_path)
trainer.save_model(out_model_path)
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")