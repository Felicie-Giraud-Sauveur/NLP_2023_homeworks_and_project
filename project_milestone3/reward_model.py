###########
## model ##
###########

import re
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, DebertaV2Config

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomRewardModelTokenizer:
    """Custom the tokenizer of the OpenAssistant Deberta model
    """

    def __init__(self, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
        self.device = device

    def extract_question_answer(self, chat):
      chat_parts = re.split(r"(System|Human|Assistant):\s*", chat)[1:]
      question = ""
      answer = ""
      for i in range(0, len(chat_parts), 2):
        role = chat_parts[i].lower()
        text = chat_parts[i+1].strip()
        if role in ["system", "human"]:
         question += text
        if role == "assistant":
          answer += text
      return question, answer

    def __call__(self, chats, *args, **kwargs):
      questions = []
      answers = []
      for chat in chats:
        question, answer = self.extract_question_answer(chat)
        questions.append(question)
        answers.append(answer)
      inputs = self.tokenizer.batch_encode_plus(list(zip(questions, answers)), **kwargs)
      # Move the encoded inputs to the desired device
      inputs = {k: v.to(self.device) for k, v in inputs.items()}
      return inputs
    
    def decode(self, encodes, **kwargs):
      decoded_texts = []
      for enc in encodes:
        decoded_text = self.tokenizer.decode(enc, **kwargs)
        decoded_texts.append(decoded_text)
      return decoded_texts



class CustomRewardModelConfig(DebertaV2Config):
  """Config class for a custom HuggingFace model.
  """
  model_type = "CustomRewardModel"


class CustomRewardModel(PreTrainedModel):
  """Custom the OpenAssistant Deberta model
  """

  config_class = CustomRewardModelConfig

  def __init__(self, config):
    super().__init__(config)
		
    self.tokenizer = CustomRewardModelTokenizer()
    self.deberta = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
    
    self.sigmoid = nn.Sigmoid()

  def forward(self, bad_inputs, good_inputs):

    bad_input_ids = bad_inputs['input_ids']
    bad_attention_mask = bad_inputs['attention_mask']
    bad_token_type_ids = bad_inputs['token_type_ids']

    good_input_ids = good_inputs['input_ids']
    good_attention_mask = good_inputs['attention_mask']
    good_token_type_ids = good_inputs['token_type_ids']

    bad_outputs = self.deberta(input_ids=bad_input_ids, attention_mask=bad_attention_mask, token_type_ids=bad_token_type_ids)
    good_outputs = self.deberta(input_ids=good_input_ids, attention_mask=good_attention_mask, token_type_ids=good_token_type_ids)
    
    bad_scores = bad_outputs.logits
    good_scores = good_outputs.logits
    
    return self.sigmoid(good_scores-bad_scores)
  

  def get_score(self, chats):
    
    # Encode the batch using the tokenizer
    encoded = self.tokenizer(chats, return_tensors="pt", truncation=True, padding=True, max_length=500)
    
    # Get predictions from the model
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']

    outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    scores = outputs.logits
    
    return scores


  def get_rewards(self, demonstrations):
      rewards = []
      for pair in demonstrations:
          scores_chosen = self.get_score([pair['chosen']])
          scores_reject = self.get_score([pair['rejected']])
          rewards.append({'chosen': scores_chosen.item(), 'rejected': scores_reject.item()
          })
      return rewards
