import os
from io import open
import torch
import json


class Dictionary(object):
    def __init__(self, vocab_dict):
        self.word2idx = vocab_dict
        self.idx2word = {v: k for k, v in vocab_dict.items()}

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, vocab_path, data_path):
        with open(vocab_path, "r") as file:
            vocab_dict = json.load(file)
        self.data_path = data_path
        self.dictionary = Dictionary(vocab_dict)
        if os.path.isfile(os.path.join(self.data_path, "")):
            self.train = torch.load(os.path.join(self.data_path, ""))
        else:
            self.train = self.tokenize(os.path.join(self.data_path, ""))

        if os.path.isfile(os.path.join(self.data_path, "")):
            self.valid = torch.load(os.path.join(self.data_path, ""))
        else:
            self.valid = self.tokenize(os.path.join(self.data_path, ""))

        if os.path.isfile(os.path.join(self.data_path, "")):
            self.test = torch.load(os.path.join(self.data_path, ""))
        else:
            self.test = self.tokenize(os.path.join(self.data_path, ""))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = list(line.strip().replace(" ", "|"))
                words.append("</s>")
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
            # BOS를 매번 쓰느니 앞에 EOS 하나 붙혀주는것 만으로도 완벽하게 처리 가능함.
            eos_id = torch.Tensor([self.dictionary.word2idx["</s>"]])
            ids = torch.cat([eos_id, ids])
            ids = ids.type(torch.int64)
            pt_path = path.split(".")[0]
            torch.save(ids, pt_path + ".pt")
        return ids
