#coding=utf-8

from feature import feature_main
from model import model_main, baseline_model
from predict import validate

def train():
    # feature_main()
    model_main()
    cls_model = baseline_model()
    cls_model.load_model()
    # predict_main(cls_model)
    validate(cls_model)

if __name__ == '__main__':
    train()