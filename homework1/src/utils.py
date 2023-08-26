from torch.utils.data import Dataset
import datasets
import torch
import torch.nn as nn


class RNNDataset(Dataset):
    def __init__(self,
                 dataset: datasets.arrow_dataset.Dataset,
                 max_seq_length: int
                 ):
        self.max_seq_length = max_seq_length
        self.prepared_dataset = dataset.map(self.start_stop_pad)
        self.vocab = self.get_dataset_vocabulary(self.prepared_dataset)
        self.data = self.prepared_dataset["text"]        

        # defining a dictionary that simply maps tokens to their respective index in the embedding matrix
        self.word_to_index = {w:i for i,w in enumerate(self.vocab)}
        self.index_to_word = {i:w for i,w in enumerate(self.vocab)}
        
        self.pad_idx = self.word_to_index["<pad>"]
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):    
        
        token_list = self.data[idx].split()
        # having a fallback to <unk> token if an unseen word is encoded.
        token_ids = [self.word_to_index.get(word, self.word_to_index["<unk>"]) for word in token_list]
        
        return torch.tensor(token_ids)

    
    def start_stop_pad(self, txt):
        """ Add <start>, <stop> and <pad> to the sentence"""
        modified_txt = txt
        modified_txt["text"] = "<start> " + modified_txt["text"] + " <stop>"
        modified_txt_split = modified_txt["text"].split()
        modified_txt_split.extend(["<pad>"] * (self.max_seq_length+2 - len(modified_txt_split)))
        modified_txt["text"] = " ".join(modified_txt_split)
        return modified_txt
    
    def get_dataset_vocabulary(self, dataset: datasets.arrow_dataset.Dataset):
        """Get the vocabulary of the dataset"""
        vocab = sorted(set(" ".join([txt["text"] for txt in dataset]).split()))
        return vocab

    

    

class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_dim,
                 hidden_dim,
                 num_layers, 
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate 

        # pass embedding weights if exist
        if embedding_weights is not None:
            self.embeddings = nn.Embedding.from_pretrained(embedding_weights)
            self.embeddings.weight.requires_grad = not freeze_embeddings
            pass

        else:  # train from scratch embeddings
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.embeddings.weight.requires_grad = not freeze_embeddings
            pass

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True
                            )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.fc = nn.Linear(self.hidden_dim, out_features=self.vocab_size)

        
    def forward(self, input_id):
        x = self.embeddings(input_id)
        x = self.dropout(x)
        x, h = self.lstm(x)
        x = self.fc(x)
        return x[:, :-1, :].permute(0, 2, 1), input_id[:, 1:]

    


class EncoderRNN(nn.Module):
    def __init__(self, input_vocab, embedding_dim, hidden_dim, device):
        super(EncoderRNN, self).__init__()
        self.vocab = input_vocab
        self.vocab_size = len(self.vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.rnn = nn.GRU(input_size=self.embedding_dim,           
                      hidden_size=self.hidden_dim,         
                      num_layers=1,                                  
                      batch_first=True,               
                      bidirectional=False,
                      )        

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(8, 1, self.embedding_dim)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.randn(1, 8, self.hidden_dim, device=self.device)    

    
    
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, output_vocab, hidden_dim, max_length, device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.vocab = output_vocab
        self.vocab_size = len(self.vocab)
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)      
        self.dropout = nn.Dropout(self.dropout_p)

        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        self.rnn = nn.GRU(input_size=self.hidden_dim,           
                          hidden_size=self.hidden_dim,         
                          num_layers=1,                                  
                          batch_first=True,               
                          bidirectional=False,
                          ) 
        
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)

        
    def forward(self, input, hidden, encoder_outputs):
        
        batch_output = torch.zeros(8, self.vocab_size, device=self.device)
        batch_hidden = torch.zeros(1, 8, self.hidden_dim, device=self.device)
        
        for b in range(8):
            
            b_input = input[b,:]
            b_hidden = hidden[:,b,:].view(1,1,-1)
            b_encoder_outputs = encoder_outputs[b,:,:]
            
            embedded = self.embedding(b_input).view(1, 1, -1)
            embedded = self.dropout(embedded)

            attn_weights = self.attn(torch.cat((embedded[0], b_hidden[0]), 1))
            attn_weights = nn.functional.softmax(attn_weights, dim=1)

            attn_applied = torch.bmm(attn_weights.unsqueeze(0), b_encoder_outputs.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)

            output = nn.functional.relu(output)
            output, b_hidden = self.rnn(output, b_hidden)
            output = nn.functional.log_softmax(self.out(output[0]), dim=1)
            
            batch_output[b,:] = output
            batch_hidden[:,b,:] = b_hidden
        
        return batch_output, batch_hidden

    def initHidden(self):
        return torch.randn(1, 8, self.hidden_dim, device=self.device)    
    





class EncoderDecoder(nn.Module):
    def __init__(self, hidden_dim, input_vocab, output_vocab, embedding_dim, max_length, device, rnn_word_to_idx, criterion):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = EncoderRNN(input_vocab, embedding_dim, hidden_dim, device)
        self.decoder = AttnDecoderRNN(output_vocab, hidden_dim, max_length, device, dropout_p=0.1)
        
        self.max_length = max_length
        self.device = device
        self.rnn_word_to_idx = rnn_word_to_idx
        self.criterion = criterion

    def forward(self, inputs):
        
        context1 = inputs[0]
        reference2 = inputs[1]
        
        loss = 0
        
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(8, self.max_length, self.encoder.hidden_dim, device=self.device)
        
        for ei in range(context1.size(1)):
            encoder_output, encoder_hidden = self.encoder(context1[:, ei], encoder_hidden)
            encoder_outputs[:, ei] = encoder_output.view(8, self.encoder.hidden_dim)
        
        decoder_input = torch.full((8, 1), self.rnn_word_to_idx["<start>"], device=self.device)
        decoder_hidden = encoder_hidden
        
        for di in range(reference2.size(1)):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(8,-1)
            
            loss += self.criterion(decoder_output, reference2[:, di])
            #if decoder_input.item() == self.rnn_word_to_idx["<stop>"]:
                #break
    
        return loss