import random
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import wordnet, stopwords


# ========================== Synonym Replacement ========================== #

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)


def synonym_replacement(sentence, n):   
    """Replace up to n random words in the sentence with their synonyms."""
    
    # Get list of words
    words = sentence.split()    
    
    # Get index of words that we can replace
    # no stopwords should be replaced
    stop_words = stopwords.words('english')
    idx_word_list = [i for i in range(0, len(words)) if words[i].lower() not in stop_words]

    # We want to replace until the number of replacement gets to n or all the words have been replaced
    n = min(n, len(idx_word_list))
    
    for r in range(n):
        
        # Choose index of the word we want to replace and remove it for futur choices
        idx_toreplace = random.choice(idx_word_list)
        idx_word_list.remove(idx_toreplace)
        
        # Replace the word by its synonym
        list_syn = get_synonyms(words[idx_toreplace])
        try:
            words[idx_toreplace] = random.choice(list_syn)
        except:
            pass
            #print("/!\ For synonym_replacement: the word '{}' was not replaced because it has no synonym. /!\ \n".format(words[idx_toreplace]))
    
    # Return a new sentence after all the replacement
    new_sentence = " ".join(words)

    return new_sentence



# ========================== Random Deletion ========================== #

def random_deletion(sentence, p, max_deletion_n):
    """Randomly delete words with probability p."""

    words = sentence.split()
    max_deletion_n = min(max_deletion_n, len(words)-1)
    
    new_sentence = []
    nb_deletion = 0
    
    # Obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return sentence
 
    # Randomly delete words with probability p
    
    # Iterate through all the words and determine whether each of them should be deleted
    # We can delete at most `max_deletion_n` words
    for i in range(len(words)):
        
        if (random.random()>p) | (nb_deletion==max_deletion_n):
            new_sentence.append(words[i])
        else:
            nb_deletion += 1
    
    # Return the new sentence after deletion
    new_sentence = " ".join(new_sentence)
    
    return new_sentence


# ========================== Random Swap ========================== #

def swap_word(sentence):
    """Randomly swap two words in the sentence."""
    
    words = sentence.split()
    
    if len(words) <= 1:
        return sentence
    
    # Randomly swap two words in the sentence
    
    # Randomly get two indices
    random_idxs = random.sample(range(0, len(words)), 2)
    random_idx_1, random_idx_2 = random_idxs[0], random_idxs[1]
    
    # Swap two tokens in these positions
    new_sentence = words.copy()
    new_sentence[random_idx_1] = words[random_idx_2]
    new_sentence[random_idx_2] = words[random_idx_1]
    
    # Return the new sentence after swap
    new_sentence = " ".join(new_sentence)

    return new_sentence


# ========================== Random Insertion ========================== #

def random_insertion(sentence, n):
    """Apply add_word n times to the sentence."""
    
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    new_sentence = ' '.join(new_words)
    return new_sentence


def add_word(new_words):
    """Randomly choose one synonym and insert it into the word list."""

    # Get a synonym word of one random word from the word list
    synonyms = []
    word_choice = new_words.copy()
    while (len(synonyms)==0) & (len(word_choice)!=0):
        random_word = random.choice(word_choice)
        synonyms = get_synonyms(random_word)
        word_choice.remove(random_word)
    
    if len(synonyms)!=0:
        random_synonym = random.choice(synonyms)
    
        # Insert the selected synonym into a random place in the word list
        new_words.insert(random.randint(0, len(new_words)), random_synonym)
