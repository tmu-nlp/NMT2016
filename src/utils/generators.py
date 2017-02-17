import random
random.seed(10)

def batch(generator, mini_batch_size, deca_batch_size): 
    
    mini_batch = []
    deca_batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        mini_batch.append(l)
        if len(mini_batch) == mini_batch_size: 
            if not is_tuple:
                yield mini_batch
                mini_batch = []
            else: 
                deca_batch.append(mini_batch) 
                if len(deca_batch) == deca_batch_size:
                    random.shuffle(deca_batch)
                    for batch in deca_batch:
                        yield tuple(list(x) for x in zip(*batch))
                    deca_batch = []
                mini_batch = []
    if mini_batch:
        if not is_tuple:
            yield mini_batch
        else:
            random.shuffle(deca_batch)
            for batch in deca_batch:
                yield tuple(list(x) for x in zip(*batch))


def sorted_parallel(generator1, generator2, pooling, order=0):
    gen1 = batch(generator1, pooling, None) 
    gen2 = batch(generator2, pooling, None)
    new_gen = list(zip(gen1, gen2))
    random.shuffle(new_gen)
    for batch1, batch2 in new_gen:
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])): 
            yield x


def word_list(filename, corpus_type="src"):
    with open(filename) as fp:
        for l in fp:
            l = l.strip().split()
            if corpus_type == "src":
                l.insert(0, "<s>")
            l.append("</s>")
            yield l

