import sys
import datetime
from argparse import ArgumentParser
from itertools import zip_longest

def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()


def fill_batch(batch, token='__NULL__'):
    return list(zip(*zip_longest(*batch, fillvalue=token)))


def arg_parser():
    p = ArgumentParser()

    p.add_argument('--mode', help='\'train\' or \'test\'')
    p.add_argument('--source', help='set source corpus file path for reading on train or test')
    p.add_argument('--target', help='set target corpus file path for reading on train or writing on test')

    p.add_argument('--model', help='set model file path for writing on train or reading on test')
    p.add_argument('--use_gpu', help='use GPU calculation')
    p.add_argument('--gpu_device', metavar='INT', type=int, help='GPU device ID to be used (default: %(default)d)')
    p.add_argument('--vocab', metavar='INT', type=int, help='vocabulary size (default: %(default)d)')
    p.add_argument('--embed', metavar='INT', type=int, help='embedding layer size (default: %(default)d)')
    p.add_argument('--hidden', metavar='INT', type=int, help='hidden layer size (default: %(default)d)')
    p.add_argument('--maxout', metavar='INT', type=int, help='maxout layer size (default: %(default)d)')
    p.add_argument('--epoch', metavar='INT', type=int, help='number of training epoch (default: %(default)d)')
    p.add_argument('--minibatch', metavar='INT', type=int, help='minibatch size (default: %(default)d)')
    p.add_argument("--pooling", type=int, help='minibatch * pooling sentences will be sorted by their length')
    p.add_argument('--generation_limit', metavar='INT', type=int, help='maximum number of words to be generated for test input (default: %(default)d)')
    p.add_argument('--optimizer', help="set SGD, Adagrad, or Adam for optimizer used for train")
    p.add_argument('--learning_rate')
    p.add_argument('--word2vec_source', default='') 
    p.add_argument('--word2vec_target', default='')

    args = p.parse_args()

    try:
        if args.mode not in ['train', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
        if args.vocab < 1: raise ValueError('set --vocab >= 1')
        if args.embed < 1: raise ValueError('set --embed >= 1')
        if args.hidden < 1: raise ValueError('set --hidden >= 1')
        if args.epoch < 1: raise ValueError('set --epoch >= 1')
        if args.minibatch < 1: raise ValueError('set --minibatch >= 1')
        if args.pooling < 1: raise ValueError('set --pooling >= 1')
        if args.generation_limit < 1: raise ValueError('set --generation-limit >= 1')
        if args.optimizer not in ['SGD', 'Adagrad', 'Adam']: raise ValueError('set mode = \'SGD\', \'Adagrad\' or \'Adam\'')
    except Exception as ex:
      p.print_usage(file=sys.stderr)
      print(ex, file=sys.stderr)
      sys.exit()

    return args
