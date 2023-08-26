import itertools
import jsonlines
import nltk
from collections import defaultdict
import string
import re


# stop_words and puntuations to be removed from consideration in the pattern extraction

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('uh')

puncs = string.punctuation


# Function for word_pair_extraction

def word_pair_extraction(prediction_files, tokenizer):
    '''
    Extract all word pairs (word_from_premise, word_from_hypothesis) from input as features.
    
    INPUT: 
      - prediction_files: file path for all predictions
      - tokenizer: tokenizer used for tokenization
    
    OUTPUT: 
      - word_pairs: a dict of all word pairs as keys, and label frequency of values. 
    '''
    word_pairs = defaultdict(lambda: [0, 0, 0])
    label_to_id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    pairs_pred_files = []
    
    for pred_file in prediction_files:
        with jsonlines.open(pred_file, "r") as reader:
            for pred in reader.iter():
                
                # Tokenize the text with 'tokenizer'
                p_tokens = tokenizer.tokenize(pred["premise"])
                h_tokens = tokenizer.tokenize(pred["hypothesis"])
                
                # Get list of unique pair words
                list_pairs = list(itertools.product(p_tokens, h_tokens))
                list_pairs = [tuple(sorted(t)) for t in list_pairs]
                list_pairs = [t for t in (set(tuple(i) for i in list_pairs))]
                
                # Remove pairs with stop_words or puncs or ## pattern
                list_pairs = [tup for tup in list_pairs if not any(i in tup for i in stop_words)]
                list_pairs = [tup for tup in list_pairs if not any(i in tup for i in puncs)]
                list_pairs = [tup for tup in list_pairs if not ((re.compile('##.*').search(tup[0])) or (re.compile('##.*').search(tup[1])))]
                
                # Remove pairs with same words in premise and hypothesis
                list_pairs = [tup for tup in list_pairs if (tup[0]!=tup[1])]
                
                # Put each pair in alphabetical order
                list_pairs = [tuple(sorted(tup)) for tup in list_pairs]
                
                # Get id of the prediction label
                label_id = label_to_id[pred["prediction"]]
                
                # Count predictions for each paired words as values
                for pair in list_pairs:
                    word_pairs[pair][label_id] += 1
                
                # Save pairs with prediction
                pairs_pred_files.append({"premise": pred["premise"], "hypothesis": pred["hypothesis"], "pairs": list_pairs, "domain": pred["domain"], "label": pred["label"], "prediction": pred["prediction"]})
    
    return word_pairs, pairs_pred_files
