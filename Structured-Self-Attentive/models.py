from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os


class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['ninp'])
        self.bilstm = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
        self.init_weights()
        # self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        # if os.path.exists(config['word-vector']):
        #     print('Loading word vectors from', config['word-vector'])
        #     vectors = torch.load(config['word-vector'])
        #     assert vectors[2] >= config['ninp']
        #     vocab = vectors[0]
        #     vectors = vectors[1]
        #     loaded_cnt = 0
        #     for word in self.dictionary.word2idx:
        #         if word not in vocab:
        #             continue
        #         real_id = self.dictionary.word2idx[word]
        #         loaded_id = vocab[word]
        #         self.encoder.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
        #         loaded_cnt += 1
        #     print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        # print("embed_size : ", emb.size())
        outp,(final_hidden, final_cell) = self.bilstm(emb, hidden)
        # print("bilstm_output_size : ", outp.size())
        # print("hiddens, cell size : ", final_hidden.size(), final_cell.size())
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)      # da(attention_dimension)-by-2u (bilstm hidden)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dictionary = config['dictionary']
        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        # print("input_size : ", inp.size())
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid]
        # print("bilstm_forward_output_size : ", size)
        compressed_embeddings = outp.reshape(-1, size[2])  # [bsz*len, nhid*2]
        # print("compressed_embeddings_size : ", compressed_embeddings.size())
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  
        # print("inp_size : ", inp.size()) # [bsz, len]
        transformed_inp = transformed_inp.reshape(size[0], 1, size[1])  # [bsz, 1, len]
        # print("transformed_inp_size : ", transformed_inp.size())
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]
        # print("concatenated_inp_size : ",concatenated_inp.size())

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        # print("hbar size : ", hbar.size())
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        # print("alphas size : ", alphas.size())
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        # print("penalized_alphas size : ", penalized_alphas.size())
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        # print("softmax alpha size : ", alphas.size())
        # print("softmax sum : ", torch.sum(alphas,dim=1))
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        # print("final alphas size : ", alphas.size())
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)


class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        if config['pooling'] == 'mean' or config['pooling'] == 'max':
            self.encoder = BiLSTM(config)
            self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
        elif config['pooling'] == 'all':
            self.encoder = SelfAttentiveEncoder(config)
            self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.dictionary = config['dictionary']
#        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        # print("attention complete size : ", outp.size())
        outp = outp.view(outp.size(0), -1)
        # print("attention complete size : ", outp.size())
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        # print("pred size : ", pred.size())
        if type(self.encoder) == BiLSTM:
            attention = None
        return pred, attention

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]
