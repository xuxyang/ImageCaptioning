import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_dim = hidden_size
        self.beam_size = 20
        self.candidate_word_count = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def init_hidden(self, batch_size):
         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        long_state_init = torch.randn(1, batch_size, self.hidden_dim)
        short_state_init = torch.randn(1, batch_size, self.hidden_dim)
        self.hidden = (long_state_init.to(self.device), short_state_init.to(self.device))

    def init_hidden_beam_search(self, batch_size):
        long_state_init = torch.randn(1, batch_size, self.hidden_dim)
        short_state_init = torch.randn(1, batch_size, self.hidden_dim)
        self.hidden = [(long_state_init.to(self.device), short_state_init.to(self.device)) for i in range(self.beam_size)]

    def init_best_sentences_candidates(self):
        # candidates is a list of tuples. Each tuple is the following structure: (past_sentence, current_vocab_idx, current_vocab_score, hidden_state_idx)
        self.candidates = list()
        # best_sentences is a list of tuples with beam_size list length. Each tuple is the following structure: (sentence_vocab_idx_list, sentence_score, hidden_state_idx)
        self.best_sentences = [list() for i in range(self.beam_size)]

    def forward(self, features, captions):
        batch_size = len(features)
        features = features.view(1, batch_size, -1) # change features shape to (1, batch_size, feature_embed_size)
        captions = captions[:,:-1] # remove the last <end>
        captions = captions.transpose(0, 1) # change shape to (sentence_len-1, batch_size)
        embeddings = self.embedding(captions)
        inputs = torch.cat((features, embeddings)) # inputs shape (sentence_len, batch_size, embed_size)
        self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.linear(lstm_out)
        return outputs.permute(1, 0, 2)

    def generate_lstm_candidates(self, lstm_input, beam_idx, candidates_count):
        " pass lstm_input through lstm and selects the top candidates_count words for each beam"
        lstm_out, self.hidden[beam_idx] = self.lstm(lstm_input, self.hidden[beam_idx])
        vocab_output = self.linear(lstm_out)
        vocab_output_score = F.log_softmax(vocab_output, dim=2)
        vocab_score_squeezed = vocab_output_score.data.squeeze()
        top_vocab_scores, top_vocab_idxs = torch.topk(vocab_score_squeezed, candidates_count)
        for vocab_idx in range(candidates_count):
            self.candidates.append((self.best_sentences[beam_idx], top_vocab_idxs[vocab_idx].item(), top_vocab_scores[vocab_idx].item(), beam_idx))



    def update_best_sentences(self):
        " selects top beam_size best sentences according to total score of the sentence"
        candidate_scores = torch.tensor([]) # candidate_scores contains all the candidate sentences' sentence scores 
        if len(self.candidates[0][0]) == 0: # first word in the sentence, sentence score is just the current word score
            candidate_scores = torch.tensor([candicate[2] for candicate in self.candidates])
        else: # Non-first word in the sentence, sentence score is the sum of past sentence score and current word score
            candidate_scores = torch.tensor([candicate[2] + candicate[0][1] for candicate in self.candidates])
        
        top_scores, top_idxs = torch.topk(candidate_scores, self.beam_size)
        for idx in range(self.beam_size):
            candicate = self.candidates[top_idxs[idx]]
            sentence_vocab_idxs = list()
            if len(candicate[0]) > 0:
                sentence_vocab_idxs.extend(candicate[0][0])
            sentence_vocab_idxs.append(candicate[1])
            self.best_sentences[idx] = (sentence_vocab_idxs, top_scores[idx].item(), candicate[3])


    def sample(self, inputs, max_len = 20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) using beam search"
        batch_size = inputs.shape[1]
        self.init_hidden_beam_search(batch_size)
        self.init_best_sentences_candidates()

        for i in range(max_len):
            self.candidates = list()
            if i == 0:
                self.generate_lstm_candidates(inputs, 0, self.beam_size)
            else:
                # update self.hidden to the selected best sentences' hidden state
                org_hidden = [state for state in self.hidden]
                for beam_idx in range(self.beam_size):
                    self.hidden[beam_idx] = org_hidden[self.best_sentences[beam_idx][2]]

                for beam_idx in range(self.beam_size):
                    vocab_idx = self.best_sentences[beam_idx][0][-1]
                    lstm_input = self.embedding(torch.tensor([[vocab_idx]], dtype=torch.long, device=self.device))
                    self.generate_lstm_candidates(lstm_input, beam_idx, self.candidate_word_count)

            self.update_best_sentences()
            
            
        #outputs = [self.best_sentences[i][0] for i in range(self.beam_size)]
        outputs = self.best_sentences[0][0]

        return outputs

