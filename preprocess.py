import os
import re
import sentencepiece as spm
from config import Config


def preprocess(hparams):
    data_path = "./data/de-en"
    train_path_en = os.path.join(data_path, "train.tags.de-en.en")
    train_path_de = os.path.join(data_path, "train.tags.de-en.de")
    val_path_en = os.path.join(data_path, "IWSLT16.TED.tst2013.de-en.en.xml")
    val_path_de = os.path.join(data_path, "IWSLT16.TED.tst2013.de-en.de.xml")
    test_path_en = os.path.join(data_path, "IWSLT16.TED.tst2014.de-en.en.xml")
    test_path_de = os.path.join(data_path, "IWSLT16.TED.tst2014.de-en.de.xml")

    read_train = lambda x: [line.strip() for line in open(x,"r").read().split("\n") if not line.startswith("<")]
    read_val = lambda x: [re.sub("<[/]?seg[^>]*>","",line).strip() for line in open(x,"r").read().split("\n") if line.startswith("<seg")]

    train_en = read_train(train_path_en)
    train_de = read_train(train_path_de)
    val_en = read_val(val_path_en)
    val_de = read_val(val_path_de)
    test_en = read_val(test_path_en)
    test_de = read_val(test_path_de)

    os.makedirs(os.path.join(data_path, "preprocess"), exist_ok=True)
    
    def write_data(data, path):
        with open(path, "w") as f:
            f.write("\n".join(data))

    write_data(train_en, os.path.join(data_path, "preprocess/train.en"))
    write_data(train_de, os.path.join(data_path, "preprocess/train.de"))
    write_data(val_en, os.path.join(data_path, "preprocess/val.en"))
    write_data(val_de, os.path.join(data_path, "preprocess/val.de"))
    write_data(test_en, os.path.join(data_path, "preprocess/test.en"))
    write_data(test_de, os.path.join(data_path, "preprocess/test.de"))
    write_data(train_en+train_de, os.path.join(data_path, "preprocess/train"))
    
    os.makedirs(os.path.join(data_path, "bpe"), exist_ok=True)
    
    train = "--input=./data/de-en/preprocess/train --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_prefix=./data/de-en/bpe/mnt --vocab_size={} --model_type=bpe".format(hparams.vocab_size)

    spm.SentencePieceTrainer.Train(train)
    
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(data_path, "bpe/mnt.model"))
    
    def bpe_write(data, path):
        with open(path, "w") as f:
            for sentence in data:
                pieces = sp.EncodeAsPieces(sentence)
                f.write(" ".join(pieces) + "\n")
    
    bpe_write(train_en, os.path.join(data_path, "bpe/train.en.bpe"))
    bpe_write(train_de, os.path.join(data_path, "bpe/train.de.bpe"))
    bpe_write(val_en, os.path.join(data_path, "bpe/val.en.bpe"))
    bpe_write(val_de, os.path.join(data_path, "bpe/val.de.bpe"))
    bpe_write(test_en, os.path.join(data_path, "bpe/test.en.bpe"))
    bpe_write(test_de, os.path.join(data_path, "bpe/test.de.bpe"))
    
if __name__ == "__main__":
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()
    preprocess(hparams)
    
    