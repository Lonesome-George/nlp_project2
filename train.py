#coding=utf-8

from base import logger
from feature import feature_main
from model import model_main, jc_model, baseline_model
from predict import validate

def train():
    words_fetcheds = [2,3]
    topKs = [5, 15, 30, 40, 5000]
    for words_fetched in words_fetcheds:
        for topK in topKs:
            feature_main(words_fetched, topK)
            model_main()
            # cls_model = jc_model()
            cls_model = baseline_model()
            cls_model.load_model()
            # predict(cls_model)
            # logger.info('words_fetched: %d, topK: %d' % (words_fetched, topK))
            print 'words_fetched: %d, topK: %d' % (words_fetched, topK)
            validate(cls_model)

if __name__ == '__main__':
    train()