#coding=utf-8

from feature import feature_main
from model import model_main
from predict import validate

def train():
    feature_main()
    model_main()
    validate()

if __name__ == '__main__':
    train()