# CHOOSE PATH
ROOT_PATH = "."


# Imports
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Open final model and its tokenizer
tokenizer = AutoTokenizer.from_pretrained(ROOT_PATH+"/final_model")
model = AutoModelForCausalLM.from_pretrained(ROOT_PATH+"/final_model")

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)


# Open prompt data
with open(ROOT_PATH+'/data/prompts.json', encoding="utf8") as json_file:
    prompts_data = json.load(json_file)



# Inference function

def infer(prompt):

  with torch.no_grad():

    prompt = "<startofstring> " + prompt + " <bot>: "
    prompt = tokenizer(prompt, return_tensors="pt")

    input_ids = prompt["input_ids"].to(DEVICE)
    attention_mask = prompt["attention_mask"].to(DEVICE)

    output = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id, top_p=0.92, top_k=50)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output



# Answer to the prompts_data

answers = []

for entry in tqdm(prompts_data):

  guid = entry['guid']
  question = entry['question']
  choices = entry.get('choices', [])

  prompt = question
  if choices:
    prompt += f" Possible answers are: {choices}."

  model_answer = infer(question)
  bot = model_answer.split("<bot>: ")[1].strip()

  answers.append({'guid': guid, 'model_answer': bot})



# Save answers in data folder
with open(ROOT_PATH+'/data/new_answers_teamRFL.json', 'w') as f:
    json.dump(answers, f, indent=4)