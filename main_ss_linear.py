import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *

import csv

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0, help='dropout rate internal to encoder RNN')
    
    args = parser.parse_args()
    return args


# Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
# returns the associated logical form.
class NearestNeighborSemanticParser(object):
    # Take any arguments necessary for parsing
    def __init__(self, training_data):
        self.training_data = training_data

    # decode should return a list of k-best lists of Derivations. A Derivation consists of the underlying Example,
    # a probability, and a tokenized output string. If you're just doing one-best decoding of example ex and you
    # produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
    def decode(self, test_data):
        # Find the highest word overlap with the test data
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    
    def __init__(self, model_enc, model_dec, model_input_emb, model_output_emb, input_indexer, output_indexer, reverse_input, max_output_len):
        self.encoder = model_enc
        self.decoder = model_dec
        self.input_embeddings = model_input_emb
        self.output_embeddings = model_output_emb
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.reverse_input = reverse_input
        self.max_pred_len = max_output_len
    
    def decode(self, test_data):
        self.encoder.eval()
        self.decoder.eval()
        self.input_embeddings.eval()
        self.output_embeddings.eval()
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)
        SOS_idx = torch.tensor(output_indexer.index_of(SOS_SYMBOL)).unsqueeze(0)
        SOS_embed = self.output_embeddings.forward(SOS_idx).unsqueeze(0)
        EOS_idx = torch.tensor(output_indexer.index_of(EOS_SYMBOL)).unsqueeze(0) 
        num_examples = len(test_data)
        preds = []
        print("In decode") 
        for idx in range(num_examples):
            print(idx)
            out_toks = []
            inp_ex = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0)
            inp_len = torch.from_numpy((np.array(len(test_data[idx].x_tok)))).unsqueeze(0)
            enc_output_each_word, enc_context_mask, enc_final_states = encode_input_for_decoder(inp_ex, inp_len, self.input_embeddings, self.encoder)
            class_scores, dec_hid = self.decoder.forward(SOS_embed, torch.tensor(1).unsqueeze(0), enc_final_states, enc_output_each_word)
            dec_hid = (dec_hid[0].unsqueeze(0),dec_hid[1].unsqueeze(0))
            out_idx = torch.max(class_scores, dim = 1)[1]
            i = 0
            while out_idx != EOS_idx and i<=self.max_pred_len:
                i += 1
                out_toks.append(self.output_indexer.get_object(out_idx.item()))
                out_emb = self.output_embeddings.forward(out_idx).unsqueeze(0)
                class_scores, dec_hid = self.decoder.forward(out_emb, torch.tensor(1).unsqueeze(0), dec_hid, enc_output_each_word)
                dec_hid = (dec_hid[0].unsqueeze(0),dec_hid[1].unsqueeze(0))
                out_idx = torch.max(class_scores, dim = 1)[1]
            preds.append([Derivation(test_data[idx], 1.0, out_toks)])
        
        return preds

# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


# Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
# inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len) #array of word indices for each sentence
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    
    #model_dec = DecoderCell(args.input_dim, args.hidden_size, len(output_indexer), dropout=0)
    model_attn_dec = AttentionDecoderCell(args.input_dim, args.hidden_size, len(output_indexer), input_max_len, dropout=0)
    
    SOS_idx = torch.tensor(output_indexer.index_of(SOS_SYMBOL)).unsqueeze(0)
    EOS_idx = output_indexer.index_of(EOS_SYMBOL)
    num_examples = all_train_input_data.shape[0]
    #all_train_input_data.shape - 480 x 19
    #all_train_output_data.shape - 480 x 65

    #params = list(model_dec.parameters()) + list(model_enc.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    params = list(model_attn_dec.parameters()) + list(model_enc.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = optim.Adam(params, lr = args.lr)
    # output_max_len = 65
    iteration = 0
    #Scheduled Sampling - Linear Decay
    k = 0.999
    c = 4e-5
    epsilon = 1
    
    with open('../NLP_lin_n1.csv','w') as ooo:
        writer = csv.writer(ooo,delimiter = ',')
        writer.writerow([' '])
        writer.writerow([' ', "k=0.999, exponential decay"])
        writer.writerow([' ', "Epoch", "Exact", "Token", "Denotation"]) 

    start_decay = False
    for epoch in range(args.epochs):
        print("epoch:", epoch)

        ex_indices = [i for i in range(num_examples)]
        random.shuffle(ex_indices) 

        for idx in ex_indices:
           
            loss = torch.autograd.Variable(torch.FloatTensor([0]))
            criterion = nn.CrossEntropyLoss()
            #model_dec.zero_grad()
            model_attn_dec.zero_grad()
            model_enc.zero_grad()
            model_input_emb.zero_grad()
            model_output_emb.zero_grad()
            
            inp_ex = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_len = torch.from_numpy((np.array(len(train_data[idx].x_tok)))).unsqueeze(0)
            gold_len = len(train_data[idx].y_indexed)
            enc_output_each_word, enc_context_mask, enc_final_states = encode_input_for_decoder(inp_ex, inp_len, model_input_emb, model_enc)

            SOS_embed = model_output_emb.forward(SOS_idx).unsqueeze(0)
            class_scores, dec_hid = model_attn_dec.forward(SOS_embed, torch.tensor(1).unsqueeze(0), enc_final_states, enc_output_each_word)
            
            for gold_idx in train_data[idx].y_indexed:
                loss += criterion(class_scores, torch.LongTensor([gold_idx]))

                if gold_idx == EOS_idx:
                    break

                dec_hid = (dec_hid[0].unsqueeze(0),dec_hid[1].unsqueeze(0))
                prev_idx = torch.max(class_scores, dim = 1)[1]
                
                if start_decay == True:
                    iteration += 1
                    epsilon = max(epsilon, k - iteration)
                    r = random.random()
                    if r <= epsilon:
                        inp_emb = model_output_emb.forward(torch.tensor(gold_idx).unsqueeze(0)).unsqueeze(0)
                    else:
                        inp_emb = model_output_emb.forward(prev_idx).unsqueeze(0)
                else:
                    inp_emb = model_output_emb.forward(torch.tensor(gold_idx).unsqueeze(0)).unsqueeze(0)
                     
                class_scores, dec_hid = model_attn_dec.forward(inp_emb, torch.tensor(1).unsqueeze(0), dec_hid, enc_output_each_word)
                    
            loss.backward()                
            optimizer.step()
            
        if epoch % 1 == 0:           
            oml = output_max_len*3
            s=Seq2SeqSemanticParser(model_enc,model_attn_dec,model_input_emb,model_output_emb,input_indexer,output_indexer,args.reverse_input,oml)
            exact, token_level, denotation =  evaluate(test_data, s)
            with open('../NLP_lin_n1.csv','a') as ooo:
                writer = csv.writer(ooo,delimiter = ',')

                if denotation > 50 and start_decay == False: 
                    start_decay = True
                    writer.writerow([' ', "start", "decay","next","epoch"])

                writer.writerow([' ', epoch, exact, token_level, denotation])

    return Seq2SeqSemanticParser(model_enc, model_attn_dec, model_input_emb, model_output_emb, input_indexer, output_indexer, args.reverse_input, output_max_len * 3)

# Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
# every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
# executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None):
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % selected_derivs[i].y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i].y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i].y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()
    
    return (float(num_exact_match)/len(test_data)) *100, (float(num_tokens_correct)/total_tokens)*100, (float(num_denotation_match)/len(test_data))*100

def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s\n" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        evaluate(dev_data_indexed, decoder)
    #print(dev_data_indexed[0])
    #print(test_data_indexed[0])
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")
