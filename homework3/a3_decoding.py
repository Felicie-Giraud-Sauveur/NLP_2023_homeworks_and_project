
import torch
from typing import Any, Dict
from a3_utils import *

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class GreedySearchDecoderForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def search(
        self,
        inputs: dict,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using greedy decoding. 
        This means that we always pick the next token with the highest score/probability.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1,generated_seq_len)
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
        # For greedy decoding, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
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
                
            # Get the next_token_id
            outputs = self.model(input_ids=model_inputs['input_ids'],
                                 attention_mask=model_inputs['attention_mask'],
                                 decoder_input_ids=model_inputs['decoder_input_ids'],
                                 decoder_attention_mask=model_inputs['decoder_attention_mask'])
            logits = outputs.logits
            new_token_id = logits.argmax(dim=2)[0, -1].item()
            
            # Add the new_token_id to the result
            gen = torch.cat((gen, torch.tensor([[new_token_id]]).to(device='cuda')), dim=1) 
            
            # Early stopping
            if new_token_id==self.tokenizer.eos_token_id:
                break
        
        return gen
             
             
            
             


class BeamSearchDecoderForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def search(
        self,
        inputs,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
    ) -> dict: 
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using beam search. 
        This means that we sample the next token according to the best conditional 
        probabilities of the next beam_size tokens.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1 and of beam size > 1 
            (1=greedy search, but you don't have to handle it)
        
        It also include a length_penalty variable that controls the score assigned to a long generation.
        Implemented by exponiating the length of the decoder inputs to this value. 
        This is then used to divide the score which can be calculated as the sum of the log probabilities so far.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (_type_): the tokenized input dictionary returned by the T5 tokenizer
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional): 
                exponential penalty to the length that is used with beam-based generation. 
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. 
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor) 
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence, 
                          you can fill the rest of the tensor row with the pad token ID
                    - "scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs, 
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # Given a probability distribution over the possible next tokens and 
        # a beam width (here num_beams), needs to keep track of the most probable 
        # num_beams candidates.
        # You can do so by keeping track of the sum of the log probabilities of 
        # the best num_beams candidates at each step.
        # Then recursively repeat this process until either:
        #   - you reach the end of the sequence
        #   - or you reach max_length
        #
        # For beam search, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to implement the length penalty
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
        
        # Initialization
        
        list_model_inputs = [self.prepare_next_inputs(model_inputs=inputs, use_cuda=True)]
        list_sequences = torch.tensor([[list_model_inputs[0]['decoder_input_ids']]]).to(device='cuda')
        list_scores = [0]
        
        # Find next tokens
        
        for i in range(1, max_new_tokens+1):
            
            tmp_model_inputs = []
            tmp_sequences = []
            tmp_scores = []
            tmp_scores_penalized = []
        
            for num_seq in range(len(list_sequences)):

                if list_sequences[num_seq][-1].item()==self.tokenizer.eos_token_id:

                  # early stopping
                  tmp_model_inputs = [list_model_inputs[num_seq]]
                  tmp_sequences = [list_sequences[num_seq]]
                  tmp_scores = [list_scores[num_seq]]
                  tmp_scores_penalized = [list_scores[num_seq]/len(list_sequences[num_seq])]

                else:
                  # Model output
                  outputs = self.model(input_ids=list_model_inputs[num_seq]['input_ids'],
                                      attention_mask=list_model_inputs[num_seq]['attention_mask'],
                                      decoder_input_ids=list_model_inputs[num_seq]['decoder_input_ids'],
                                      decoder_attention_mask=list_model_inputs[num_seq]['decoder_attention_mask'])
                  
                  # Probabilities
                  logits = outputs.logits
                  sftmx = torch.nn.Softmax(dim=0)
                  proba = sftmx(logits[0, -1])

                  # Keep num_beams most probable tokens
                  next_tokens_id = torch.topk(proba, num_beams).indices.tolist()
                  proba_next_tokens = torch.topk(proba, num_beams).values.tolist()
              
                  # Store temporary result
                  tmp_model_inputs += [self.prepare_next_inputs(model_inputs=list_model_inputs[num_seq], new_token_id=new_token_id, use_cuda=True) for new_token_id in next_tokens_id]
                  tmp_sequences += [torch.cat((list_sequences[num_seq], torch.tensor([new_token_id]).to(device='cuda'))) for new_token_id in next_tokens_id]
                  tmp_scores += [list_scores[num_seq]+np.log(proba_new_tokens) for proba_new_tokens in proba_next_tokens]
                  tmp_scores_penalized = [tmp_scores[i]/(tmp_sequences[i].shape[0])**length_penalty for i in range(len(tmp_scores))]

            # Keep the num_beams most probable tokens among all
            idx_tokeep = np.argpartition(np.array(tmp_scores_penalized), -num_beams)[-num_beams:]
            list_model_inputs = [tmp_model_inputs[i] for i in idx_tokeep]
            list_sequences = [tmp_sequences[i] for i in idx_tokeep]
            list_scores = [tmp_scores[i] for i in idx_tokeep]

        
        # Apply length penalization
        list_penalized_scores = [list_scores[i]/(list_sequences[i].shape[0])**length_penalty for i in range(len(list_sequences))]
        
        # Order from best scoring sequence to worst
        idx_order = np.argsort(list_penalized_scores)[::-1][:len(list_penalized_scores)]
        list_sequences = [list_sequences[i] for i in idx_order]
        list_penalized_scores = [list_penalized_scores[i] for i in idx_order]

        # if a sequence has reached end of the sentence, you can fill the rest of the tensor row with the pad token ID
        for i in range(len(list_sequences)):
          if list_sequences[i].shape[0] < max_new_tokens:
            list_sequences[i] = torch.cat((list_sequences[i], torch.tensor([self.tokenizer.pad_token_id]*(max_new_tokens-list_sequences[i].shape[0])).to(device='cuda')))
        
        # Construct dictionary
        dict_gen = {"sequences": torch.stack(list_sequences[0:num_return_sequences]),
                    "scores": list_penalized_scores[0:num_return_sequences]}
        
        
        return dict_gen


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