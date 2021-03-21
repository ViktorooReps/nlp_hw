from typing import List, Any
from random import random

from collections import defaultdict

import random
import string
import itertools
import gc
import os
import pickle
import time

import numpy as np

# TODO: implement continue learning

try:
    from score import load_dataset_fast

except ImportError:
    def load_dataset_fast(parts=None):
        return { "dev": (None, None, None) }

try:
    import resource

    def limit_memory(maxsize): 
        _, hard = resource.getrlimit(resource.RLIMIT_AS) 
        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))  

except ImportError:
    def limit_memory(maxsize):
        print("Limiting failed")
        return 


def save_obj(obj, name): 
    """Pickle obj to pickled folder"""
    with open('pickled/'+ name + '.pkl', 'wb') as f:  
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 

def load_obj(name): 
    """Unpickle object from pickled folder"""
    with open('pickled/' + name + '.pkl', 'rb') as f: 
        return pickle.load(f)

hyperparams = {
    "ngrams": 3,
    "min_freq": 10,
    "batch_size": 10000,
    "d2v_lr": 0.25,
    "emb_size": 500,
    "neg_samples": 5,
    "use_unk": False
}
systemparams = {
    "memory_limit": 12.5 * 1024 * 1024 * 1024,
    "save_preprocessed": False,
    "use_saved_preprocessed": False,
    "save_doc2vec": True,
    "use_saved_doc2vec": False
}


class Preprocessor:
    """Splits documents into tokens, creates vocabulary"""
    ENTRIES = 0 
    ID = 1
    NO_ID = -1
    UNK = "<UNK>"

    def __init__(self, ngrams=3, min_entries=5):
        self.data = {}                  # pairs (token, [entries, id])
        self.voc = set()                # vocabulary of tokens
        self.ngrams = ngrams
        self.trimmed = False            # once trimmed, freezes voc and sets token ids
        self.min_entries = min_entries  # minimum entries of tokens in vocabulary

    def process(self, doc):
        """Converts doc to list of tokens, adds tokens to voc"""
        doc = doc.lower().strip().replace("\t", "")

        for punct in string.punctuation:
            doc = doc.replace(punct, " " + punct + " ")

        splitted = list(filter(lambda a: a != "", doc.split(" ")))

        tokens = []
        for n in range(1, self.ngrams + 1):
            tokens += [tuple(splitted[j] for j in range(i, i + n)) for i in range(len(splitted) - n + 1)]
        
        if not self.trimmed:
            for token in tokens:
                if token in self.voc:
                    self.data[token][self.ENTRIES] += 1
                else:
                    self.voc.add(token)
                    self.data[token] = [1, self.NO_ID]

        return tokens

    def trim(self):
        """Trimms voc according to min_bw and min_entries, freezes voc"""
        if self.trimmed:
            print("Attempt to retrim!")
            return 0

        deleted_tokens = [] 
        for token in self.voc:
            if self.data[token][self.ENTRIES] < self.min_entries:
                deleted_tokens.append(token)

        use_unk = hyperparams["use_unk"]

        if use_unk:
            self.data[self.UNK] = [0, self.NO_ID]

        for token in deleted_tokens:
            if use_unk:
                self.data[self.UNK][self.ENTRIES] += self.data[token][self.ENTRIES]

            del self.data[token]
            self.voc.remove(token)

        gc.collect()

        if use_unk:
            self.voc.add(self.UNK)

        tok_id = 0
        for token in self.voc:
            self.data[token][self.ID] = tok_id
            tok_id += 1

        self.voc = frozenset(self.voc)
        self.trimmed = True

        total_len = len(self.voc)
        print("\nVocabulary is trimmed! Total elements: " + str(total_len))

        return total_len

    def convert(self, doc):
        """Converts doc to ids array (and trims voc if it is not trimmed)"""
        if not self.trimmed:
            self.trim()

        if hyperparams["use_unk"]:
            return [self.data[token][self.ID] if token in self.voc else self.data[self.UNK][self.ID] for token in doc]
        else:
            res = []
            for token in doc:
                if token in self.voc:
                    res.append(self.data[token][self.ID])

            return res

    def create_distribution(self):
        """Creates array representing distribution P(w) ~ U^(3/4)(w) of tokens"""
        res = np.zeros(len(self.voc), dtype=np.float32)
        norm_const = 0
        for token in self.voc:
            scaled = self.data[token][self.ENTRIES] ** (3 / 4)
            norm_const += scaled

            res[self.data[token][self.ID]] = scaled

        res /= norm_const

        return res


class Data:
    """Container for processed texts"""

    def __init__(self, preprocessor: Preprocessor, texts_list):
        tokened_list = []
        processed = 0
        for text in texts_list:
            tokened_list.append(preprocessor.process(text))
            processed += 1 

            if processed % 10000 == 0:
                print("processed: " + str(processed) + " voc size: " + str(len(preprocessor.voc)))

        preprocessor.trim()

        print("\nPreparing doc-tok index pairs...")
        
        self.doc_idxs = []
        self.tok_idxs = []

        self.total_toks = len(preprocessor.voc)
        self.total_docs = len(tokened_list)
        self.total_pairs = 0

        for doc_idx, doc in enumerate(tokened_list):
            for tok_idx in preprocessor.convert(doc):
                self.doc_idxs.append(doc_idx)
                self.tok_idxs.append(tok_idx)

                self.total_pairs += 1
        
        self.doc_idxs = np.array(self.doc_idxs, dtype=np.int32)
        self.tok_idxs = np.array(self.tok_idxs, dtype=np.int32)

        print("Finished, total pairs: " + str(self.total_pairs))

    def shuffle(self):
        indices = np.arange(self.doc_idxs.shape[0])
        np.random.shuffle(indices)

        self.doc_idxs = self.doc_idxs[indices]
        self.tok_idxs = self.tok_idxs[indices]


def batch_generator(data: Data, distrib, nb=5, batch_size=100):
    """Returns generator that generates batch = (token_idxs, docs_idxs, labels)  
    from data with nb negative samples for each positive sample using  
    distribution (distrib) of token ids  
    
    token_idxs - batch_size of token idx  
    docs_idxs  - their corresponding doc idxs  
    labels     - 0 if token tdx came from negative sampling, 1 if came from positive sampling"""
    
    print("total batches:" + str(data.total_pairs // batch_size))
    for bnum in range(data.total_pairs // batch_size):
        start_time = time.time()

        batch_start = bnum * batch_size
        batch_end = (bnum + 1) * batch_size

        neg_batch_size = batch_size * nb

        total_batch_size = batch_size + neg_batch_size

        indices = np.arange(total_batch_size)
        np.random.shuffle(indices)
        
        batch_docs = np.concatenate(
            (
                data.doc_idxs[batch_start : batch_end], 
                np.repeat(data.doc_idxs[batch_start : batch_end], nb)
            ), 
            axis=None
        )[indices]

        batch_toks = np.concatenate(
            (
                data.tok_idxs[batch_start : batch_end], 
                np.random.choice(data.total_toks, size=neg_batch_size, p=distrib)
            ), 
            axis=None
        )[indices]

        batch_labels = np.concatenate(
            (
                np.ones(batch_size, dtype=np.float32), 
                np.zeros(neg_batch_size, dtype=np.float32)
            ), 
            axis=None
        )[indices]

        gen_time = time.time() - start_time

        yield (batch_toks, batch_docs, batch_labels, gen_time)

eps = np.finfo(float).eps

def safe_sigmoid(z):
    a = sigmoid(z)
    return a - eps * a + (1 - a) * eps 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Doc2Vec:
    """Container for token and doc embeddings"""

    def __init__(self, num_docs, num_toks, emb_size=500):
        self.emb_size = emb_size
        self.doc_embs = np.random.uniform(low=-0.001, high=0.001, size=(num_docs, emb_size))
        self.tok_embs = np.random.uniform(low=-0.001, high=0.001, size=(num_toks, emb_size))

    # results for 50000 examples:
    # einsum (1.3 sec - 100, 0.74 sec - 10000) 2,5; 2,7; 4,7
    def train(self, tok_idxs, doc_idxs, labels, lr=0.25):
        Z_time = time.time()
        Z = sigmoid(np.einsum("ij,ij->i", self.tok_embs[tok_idxs], self.doc_embs[doc_idxs]))
        Z_time = time.time() - Z_time 

        batch_size = len(tok_idxs)
        
        grad_time = time.time()
        doc_grad = (labels - Z).reshape((batch_size, 1)) * self.tok_embs[tok_idxs] * lr
        tok_grad = (labels - Z).reshape((batch_size, 1)) * self.doc_embs[doc_idxs] * lr
        grad_time = time.time() - grad_time
        
        sum_time = time.time()
        self.tok_embs[tok_idxs] += tok_grad
        self.doc_embs[doc_idxs] += doc_grad
        sum_time = time.time() - sum_time

        return (Z_time, grad_time, sum_time)

    # results for 500 batches:
    # for-loop (10.4 sec - 100, 11.0 sec - 10000) 3.8; 4.0; 2.2
    def train2(self, tok_idxs, doc_idxs, labels, lr=0.25):
        total_Z_time = 0
        total_grad_time = 0
        total_sum_time = 0

        for tok_idx, doc_idx, lbl in zip(tok_idxs, doc_idxs, labels):
            Z_time = time.time()
            Z = sigmoid(self.tok_embs[tok_idx].T @ self.doc_embs[doc_idx])
            Z_time = time.time() - Z_time 

            grad_time = time.time()
            doc_grad = self.tok_embs[tok_idx] * (lbl - Z) * lr
            tok_grad = self.doc_embs[doc_idx] * (lbl - Z) * lr
            grad_time = time.time() - grad_time

            sum_time = time.time()
            self.tok_embs[tok_idx] += tok_grad
            self.doc_embs[doc_idx] += doc_grad
            sum_time = time.time() - sum_time

            total_Z_time += Z_time
            total_grad_time += grad_time
            total_sum_time += sum_time

        return (total_Z_time, total_grad_time, total_sum_time)
    

    def calculate_loss(self, tok_idxs, doc_idxs, labels):
        Z = safe_sigmoid(np.sum(self.tok_embs[tok_idxs] * self.doc_embs[doc_idxs], axis=1))
        return - np.sum(labels * np.log(Z) + (1 - labels) * np.log(1 - Z))

    def fit(self, data: Data, distrib, epochs=10):
        """Trains embeddings on given data, returns losses over epochs"""
        print("\nFitting doc2vec to doc-tok pairs")

        losses = []

        for epoch in range(epochs):
            data.shuffle()

            total_batches = 0
            total_time = 0
            total_gen_time = 0
            total_Z_time = 0
            total_grad_time = 0
            total_sum_time = 0

            first_iter = True
            res = None
            for tok_idxs, doc_idxs, labels, gen_time in batch_generator(
                data, distrib, batch_size=hyperparams["batch_size"]):

                start_time = time.time()                
                Z_time, grad_time, sum_time = self.train(tok_idxs, doc_idxs, labels, lr=hyperparams["d2v_lr"])
                secs = time.time() - start_time

                total_Z_time += Z_time
                total_grad_time += grad_time
                total_sum_time += sum_time

                total_time += secs
                total_gen_time += gen_time
                
                total_batches += 1
                if total_batches % 100 == 0:
                    loss = self.calculate_loss(tok_idxs[:100], doc_idxs[:100], labels[:100])
                    losses.append(loss)

                    print("\ntime spent generating batches: " + str(total_gen_time) + " sec")
                    print("time spent calculating: " + str(total_time) + " sec")
                    print("time spent calculating Z: " + str(total_Z_time) + " sec")
                    print("time spent calculating grad: " + str(total_grad_time) + " sec")
                    print("time spent calculating sum: " + str(total_sum_time) + " sec")
                    print("loss: " + str(loss) + " on batch " + str(total_batches))
                    total_time = 0
                    total_gen_time = 0
                    total_Z_time = 0
                    total_grad_time = 0
                    total_sum_time = 0

        return losses


def train_LR(embs, train_data, val_data):
    """Trains logistic regression classifier on train_data (doc_embs, labels)
    
    Returns (model, train_accuracy, val_accuracy)"""
    pass

def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.  
    :param train_texts: a list of texts (str objects), one str per example  
    :param train_labels: a list of labels, one label per example  
    :param pretrain_params: parameters that were learned at the pretrain step  
    :return: learnt parameters, or any object you like (it will be passed to the classify function)  
    """
    limit_memory(systemparams["memory_limit"])
    random.seed(42)
    np.random.seed(42)

    pretrain_data = pretrain_params["pretrain_data"]
    tok_distr = pretrain_params["token_distridution"]
    if systemparams["save_preprocessed"]:
        save_obj(pretrain_data, "pretrain_data")
        save_obj(tok_distr, "token_distridution")

    if systemparams["use_saved_doc2vec"]:
        d2v = load_obj("doc2vec")
    else:
        d2v = Doc2Vec(num_docs=pretrain_data.total_docs, num_toks=pretrain_data.total_toks)
        losses = d2v.fit(pretrain_data, tok_distr)
        save_obj(losses, "losses")

    if systemparams["save_doc2vec"]:
        d2v.tok_embs = None
        save_obj(d2v, "doc2vec")

    return None

def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.  
    :param texts_list: a list of list of texts (str objects), one str per example.  
        It might be several sets of texts, for example, train and unlabeled sets.  
    :return: learnt parameters, or any object you like (it will be passed to the train function)  
    """
    limit_memory(systemparams["memory_limit"])
    random.seed(42)
    np.random.seed(42)

    conc_texts = list(itertools.chain(*texts_list))
    
    if systemparams["use_saved_preprocessed"]:
        pretrain_data = load_obj("pretrain_data")
        tok_distr = load_obj("token_distridution")
    else:
        prep = Preprocessor(ngrams=hyperparams["ngrams"], min_entries=hyperparams["min_freq"])
        pretrain_data = Data(prep, conc_texts)
        tok_distr = prep.create_distribution()

    return {
        "pretrain_data": pretrain_data,
        "token_distridution": tok_distr
    }

doc_idx = 0 # considering that document order will be the same as in pretrain
def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.  
    :param texts: texts to classify  
    :param params: parameters received from train function  
    :return: list of labels corresponding to the given list of texts  
    """  
    limit_memory(systemparams["memory_limit"])
    random.seed(42)
    np.random.seed(42)

    print(len(texts))
    return ["pos"] * len(texts)
