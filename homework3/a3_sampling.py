import torch
from typing import Any, Dict
from a3_utils import *

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class TopKSamplerForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.
        It also only handles top_k => 1.
        The temperature variable that helps modulate the probability by scaling the logits.
        distribution we sample from by scaling the logits before softmax.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            top_k (int): the number of highest probability vocabulary tokens to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, scales logits before softmax
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For top-k sampling, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to implement the temperature functionality!
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        
        # To store the result (generated tokens)
        gen = torch.tensor([[]], dtype=int).to(device='cuda')
        
        for i in range(max_new_tokens):
            
            # Prepare model inputs
            if i==0:
                model_inputs = self.prepare_next_inputs(model_inputs=inputs, use_cuda=True)
                gen = torch.cat((gen, model_inputs['decoder_input_ids'].to(device='cuda')), dim=1)
            else:
                model_inputs = self.prepare_next_inputs(model_inputs=model_inputs, new_token_id=new_token_id, use_cuda=True)
                
            # Get the probabilites of next tokens (modulated by temperature)
            outputs = self.model(input_ids=model_inputs['input_ids'],
                                 attention_mask=model_inputs['attention_mask'],
                                 decoder_input_ids=model_inputs['decoder_input_ids'],
                                 decoder_attention_mask=model_inputs['decoder_attention_mask'])
            logits = outputs.logits
            sftmx = torch.nn.Softmax(dim=0)
            proba = sftmx(logits[0, -1]/temperature)

            # Keep top_k most probable tokens
            next_tokens_id_topk = torch.topk(proba, top_k).indices.tolist()

            # Sample next token id among the topk tokens
            new_token_id = random.choice(next_tokens_id_topk)
            
            # Add the new_token_id to the result
            gen = torch.cat((gen, torch.tensor([[new_token_id]]).to(device='cuda')), dim=1) 
            
            # Early stopping
            if new_token_id==self.tokenizer.eos_token_id:
                break
        
        return gen





class TopPSamplerForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p or higher.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) then sample the top scoring token.
        The temperature variable that helps modulate the probability by scaling the logits.
        distribution we sample from by scaling the logits before softmax.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, scales logits before softmax
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For top-p sampling, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to handle the edge case when top scoring tokens probability > top_p,
        #     sample that token only.
        #   - don't forget to implement the temperature functionality!
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        
        # To store the result (generated tokens)
        gen = torch.tensor([[]], dtype=int).to(device='cuda')
        
        for i in range(max_new_tokens):
            
            # Prepare model inputs
            if i==0:
                model_inputs = self.prepare_next_inputs(model_inputs=inputs, use_cuda=True)
                gen = torch.cat((gen, model_inputs['decoder_input_ids'].to(device='cuda')), dim=1)
            else:
                model_inputs = self.prepare_next_inputs(model_inputs=model_inputs, new_token_id=new_token_id, use_cuda=True)
                
            # Get the probabilites of next tokens (modulated by temperature)
            outputs = self.model(input_ids=model_inputs['input_ids'],
                                 attention_mask=model_inputs['attention_mask'],
                                 decoder_input_ids=model_inputs['decoder_input_ids'],
                                 decoder_attention_mask=model_inputs['decoder_attention_mask'])
            logits = outputs.logits
            sftmx = torch.nn.Softmax(dim=0)
            proba = sftmx(logits[0, -1]/temperature)

            # Keep top_p most probable tokens
            count_ = 0
            sum_probas = 0
            while sum_probas < top_p:
              count_ += 1
              sum_probas = sum(torch.topk(proba, count_).values.tolist())
              next_tokens_id_topp = torch.topk(proba, count_).indices.tolist()

            # Sample next token id among the top_p tokens
            new_token_id = random.choice(next_tokens_id_topp)
            
            # Add the new_token_id to the result
            gen = torch.cat((gen, torch.tensor([[new_token_id]]).to(device='cuda')), dim=1) 
            
            # Early stopping
            if new_token_id==self.tokenizer.eos_token_id:
                break
        
        return gen





def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()