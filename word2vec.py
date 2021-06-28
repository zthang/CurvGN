import logging
import os
import sys
import multiprocessing

import gensim
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    sentences=LineSentence(inp)
    bigram_transformer = gensim.models.Phrases(sentences)
    model = Word2Vec(bigram_transformer[sentences], vector_size=1433, window=10, min_count=15,sample=1e-5,negative=15,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
