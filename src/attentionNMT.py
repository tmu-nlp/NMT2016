import sys
import numpy
from chainer import Chain, Variable, cuda , optimizer, optimizers, serializer, serializers
from chainer import links as L
from chainer import functions as F
import utils.generators as gens
from utils.functions import trace, fill_batch, arg_parser
from utils.vocabulary import Vocabulary, null
from gensim.models import word2vec


class XP:
    __lib = None
    
    @staticmethod
    def set_library(args):
        if args.use_gpu:
            XP.__lib = cuda.cupy
            cuda.get_device(args.gpu_device).use()
        else:
            XP.__lib = numpy
    
    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype))
    
    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)
      
    @staticmethod
    def izeros(shape):
        return XP.__zeros(shape, XP.__lib.int32)
    
    @staticmethod
    def __nonzeros(shape, dtype, val):
        return Variable(val * XP.__lib.ones(shape, dtype=dtype))
    
    @staticmethod
    def fnonzeros(shape, val=1):
        return XP.__nonzeros(shape, XP.__lib.float32, val)
    
    @staticmethod
    def __array(array, dtype):
        return Variable(XP.__lib.array(array, dtype=dtype))
    
    @staticmethod
    def iarray(array):
        return XP.__array(array, XP.__lib.int32)
    
    @staticmethod
    def farray(array):
        return XP.__array(array, XP.__lib.float32)


class SrcEmbed(Chain):
    def __init__(self, vocab_size, embed_size):
        super(SrcEmbed, self).__init__(
             word2emb = L.EmbedID(vocab_size, embed_size),
        )
    
    def initialize_embed_w2v(self, src_vocab, word2vec_model):
        for i in range(src_vocab.__len__()):
            word = src_vocab.itos(i)
            if word == "":
                break
            if word in word2vec_model:
                vec = word2vec_model[word]
                self.word2emb.W.data[i] = vec

    def __call__(self, x):
        return F.tanh(self.word2emb(x))
   

class Encoder(Chain):
    def __init__(self, embed_size, hidden_size):
        super(Encoder, self).__init__(
            emb2hid = L.Linear(embed_size, 4 * hidden_size),
            hid2hid = L.Linear(hidden_size, 4 * hidden_size),
        )

    def __call__(self, x, xid, c, h):
        return F.lstm(c, self.emb2hid(x) + self.hid2hid(h)) 


class Attention(Chain):
    def __init__(self, hidden_size):
        super(Attention, self).__init__(
            pw = L.Linear(hidden_size, hidden_size),
            we = L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, h_list, s_list, s_len, xid_list, p):
        batch_size = p.data.shape[0]
        p = F.broadcast_to(F.reshape(p, (batch_size, 1, self.hidden_size)), (batch_size, s_len, self.hidden_size))
        new_h_list = F.reshape(h_list, (batch_size, s_len, self.hidden_size*2))
        new_s_list = F.reshape(s_list, (batch_size, s_len, self.hidden_size))
        new_s_list_reshaped4matrix_calc = F.reshape(new_s_list, (batch_size*s_len, self.hidden_size))
        p = F.reshape(p, (batch_size*s_len, self.hidden_size))
        e = F.reshape(F.softmax(F.reshape(self.we(F.tanh(new_s_list_reshaped4matrix_calc + self.pw(p))), (batch_size, s_len))), (batch_size, s_len, 1))
        ss = F.tanh(F.sum(F.scale(new_h_list, e, axis=0), axis=1))
        return ss


class Pre_Calculater(Chain):
    def __init__(self, hidden_size):
        super().__init__(
            fw = L.Linear(hidden_size, hidden_size),
            bw = L.Linear(hidden_size, hidden_size),
            )
    def __call__(self, f, b):
        return self.fw(f) + self.bw(b)


class Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, maxout_size):
        super(Decoder, self).__init__(
            word2emb = L.EmbedID(vocab_size, embed_size),
            emb2hid = L.Linear(embed_size, 4 * hidden_size),
            hid2hid = L.Linear(hidden_size, 4 * hidden_size),
            att2hid = L.Linear(hidden_size*2, 4 * hidden_size),
            emb_hid_att2maxout = L.Maxout(embed_size+hidden_size+hidden_size*2, maxout_size, 2),
            
            #emb_hid_att2maxout = L.Linear(embed_size+hidden_size+hidden_size*2, maxout_size), 
            maxout2y = L.Linear(maxout_size, vocab_size),
        )

    def initialize_embed_w2v(self, trg_vocab, word2vec_model):
        for i in range(trg_vocab.__len__()):
            word = trg_vocab.itos(i)
            if word == "":
                break
            if word in word2vec_model:
                vec = word2vec_model[word]
                self.word2emb.W.data[i] = vec

    def __call__(self, y, c, h, s):
        e = F.tanh(self.word2emb(y))
        c, h = F.lstm(c, self.emb2hid(e) + self.hid2hid(h) + self.att2hid(s))
        return self.maxout2y(F.tanh(self.emb_hid_att2maxout(F.concat((h, s, e))))), c, h


class AttentionMT(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, maxout_size):
        super(AttentionMT, self).__init__(
            emb = SrcEmbed(vocab_size, embed_size), 
            forward_enc = Encoder(embed_size, hidden_size),
            backward_enc = Encoder(embed_size, hidden_size),
            pre_calc = Pre_Calculater(hidden_size), 
            att = Attention(hidden_size),
            dec = Decoder(vocab_size, embed_size, hidden_size, maxout_size),
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def initialize_emb(self, src_vocab=None, src_word2vec='', trg_vocab=None, trg_word2vec=''):
        if src_word2vec:
            self.emb.initialize_embed_w2v(src_vocab, src_word2vec)
        if trg_word2vec:
            self.dec.initialize_embed_w2v(trg_vocab, trg_word2vec)

    def reset(self, batch_size):
        self.zerograds()
        self.x_list = []
        self.xid_list = []

    def embed(self, x):
        self.x_list.append(self.emb(x))
        self.xid_list.append(Variable(x.data.reshape(x.data.shape[0], 1)))

    def encode(self):
        src_len = len(self.x_list)
        batch_size = self.x_list[0].data.shape[0]
        zeros = XP.fzeros((batch_size, self.hidden_size))
        c = zeros
        b = zeros
        b_list = []
        for x, xid in zip(reversed(self.x_list), reversed(self.xid_list)):
            c, b = self.backward_enc(x, xid, c, b)
            b_list.insert(0, b)

        c = zeros
        a = zeros
        h_list = []
        s_list = []
        for x, xid, b in zip(self.x_list, self.xid_list, b_list): 
            c, a = self.forward_enc(x, xid, c, a)
            h_list.append(F.concat((a, b)))
            s_list.append(self.pre_calc(a,b))

        self.s_len = len(s_list)
        self.h_list = F.concat(h_list)
        self.s_list = F.concat(s_list)

        self.c = zeros
        self.h = zeros
  

    def decode(self, y):
        ss = self.att(self.h_list, self.s_list, self.s_len, self.xid_list, self.h)
        y, self.c, self.h = self.dec(y, self.c, self.h, ss)
        return y

    def save_spec(self, filename, args):
        with open(filename, 'w') as fp:
            print('vocab_size: ' + str(args.vocab), file=fp)
            print('embed_size: ' + str(args.embed), file=fp)
            print('hidden_size: ' + str(args.hidden), file=fp)
            print('maxout_size: ' + str(args.maxout), file=fp)
            print('minibatch: ' + str(args.minibatch), file=fp)
            print('pooling: ' + str(args.pooling), file=fp)
            print('optimizer: ' + str(args.optimizer), file=fp)
            print('learning_rate: ' + str(args.learning_rate), file=fp)

    @staticmethod
    def load_spec(filename):
        with open(filename) as fp:
            vocab_size = int(next(fp).split(' ')[1])
            embed_size = int(next(fp).split(' ')[1])
            hidden_size = int(next(fp).split(' ')[1])
            maxout_size = int(next(fp).split(' ')[1])
        print(vocab_size, embed_size, hidden_size, maxout_size)
        return AttentionMT(vocab_size, embed_size, hidden_size, maxout_size)


def forward(src_batch, trg_batch, src_vocab, trg_vocab, attmt, is_training, generation_limit):
    
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    trg_len = len(trg_batch[0]) if trg_batch else 0
    src_stoi = src_vocab.stoi 
    trg_stoi = trg_vocab.stoi
    trg_itos = trg_vocab.itos
    attmt.reset(batch_size)

    for l in range(src_len):
        x = XP.iarray([src_stoi(src_batch[k][l]) for k in range(batch_size)])
        attmt.embed(x)

    attmt.encode()
     
    
    t = XP.iarray([trg_stoi('<s>') for _ in range(batch_size)])
    hyp_batch = [[] for _ in range(batch_size)]

    if is_training:
        loss = XP.fzeros(())
        for l in range(trg_len):
            y = attmt.decode(t)
            t = XP.iarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)])
            loss += F.softmax_cross_entropy(y, t)
            output = cuda.to_cpu(y.data.argmax(1))
            for k in range(batch_size):
                hyp_batch[k].append(trg_itos(output[k]))
        return hyp_batch, loss
     
    else:
        while len(hyp_batch[0]) < generation_limit:
            y = attmt.decode(t)
            output = cuda.to_cpu(y.data.argmax(1))
            t = XP.iarray(output)

            for k in range(batch_size):
                hyp_batch[k].append(trg_itos(output[k]))
            if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
                break
        return hyp_batch


def train(args):
    trace('making vocabularies ...')
    src_vocab = Vocabulary.new(gens.word_list(args.source, "src"), args.vocab)
    trg_vocab = Vocabulary.new(gens.word_list(args.target, "trg"), args.vocab)
    trace('making model ...')

    attmt = AttentionMT(args.vocab, args.embed, args.hidden, args.maxout)

    if args.word2vec_source:
        attmt.initialize_emb(src_vocab, word2vec.Word2Vec.load(args.word2vec_source))
    if args.word2vec_target:
        attmt.initialize_emb(trg_vocab, word2vec.Word2Vec.load(args.word2vec_target))

    if args.use_gpu:
        attmt.to_gpu()
    if args.optimizer == "SGD":
        opt = optimizers.SGD(lr=float(args.learning_rate))
    elif args.optimizer == "Adagrad":
        opt = optimizers.AdaGrad(lr=float(args.learning_rate))
    else:
        opt = optimizers.Adam()
    opt.setup(attmt)
    opt.add_hook(optimizer.GradientClipping(5))

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        trained = 0
        gen1 = gens.word_list(args.source, "src")
        gen2 = gens.word_list(args.target, "trg")
        gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, args.pooling * args.minibatch), args.minibatch, args.pooling)
        accum_loss = 0
          
        for src_batch, trg_batch in gen3:
            src_batch = fill_batch(src_batch)
            trg_batch = fill_batch(trg_batch)
            K = len(src_batch)
            hyp_batch, loss = forward(src_batch, trg_batch, src_vocab, trg_vocab, attmt, True, 0)
            accum_loss += loss.data
            loss.backward()
            opt.update()
            
            
            for k in range(K):
                trace('epoch %3d/%3d, sample %8d' % (epoch + 1, args.epoch, trained + k + 1))
                trace('  src = ' + ' '.join([x if x != '__NULL__' else '*' for x in src_batch[k]]))
                trace('  trg = ' + ' '.join([x if x != '__NULL__' else '*' for x in trg_batch[k]]))
                trace('  hyp = ' + ' '.join([x if x != '__NULL__' else '*' for x in hyp_batch[k]]))
            trained += K
         
        trace("accum_loss = " + str(accum_loss))
         
        trace('saving model ...')
        prefix = args.model + '.%03.d' % (epoch + 1)
        src_vocab.save(prefix + '.srcvocab')
        trg_vocab.save(prefix + '.trgvocab')
        attmt.save_spec(prefix + '.spec', args)
        serializers.save_hdf5(prefix + '.weights', attmt)

    trace('finished.')

def test(args):
    trace('loading model ...')
    src_vocab = Vocabulary.load(args.model + '.srcvocab')
    trg_vocab = Vocabulary.load(args.model + '.trgvocab')
    attmt = AttentionMT.load_spec(args.model + '.spec')
    if args.use_gpu:
        attmt.to_gpu()
    serializers.load_hdf5(args.model + '.weights', attmt)
     
    trace('generating translation ...')
    generated = 0

    with open(args.target, 'w') as fp:
        for src_batch in gens.batch(gens.word_list(args.source, "src"), args.minibatch, None):
            src_batch = fill_batch(src_batch)
            K = len(src_batch)

            trace('sample %8d - %8d ...' % (generated + 1, generated + K))

            hyp_batch = forward(src_batch, None, src_vocab, trg_vocab, attmt, False, args.generation_limit)

            for hyp in hyp_batch:
                hyp.append('</s>')
                hyp = hyp[:hyp.index('</s>')]
                print(' '.join(hyp), file=fp)

            generated += K

    trace('finished.')

def main():
    args = arg_parser()
    XP.set_library(args)
    if args.mode == 'train': train(args)
    elif args.mode == 'test': test(args)

if __name__ == '__main__':
    main()

