#coding: UTF-8
from config_util import ConfigUtil
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

def get_text(config):
  message = config.get('message','input_message')
  print(message)
  text = input('>>')
  return text

def tokenize(text:str):
  tokenizer = Tokenizer()
  return tokenizer.tokenize(text, wakati=True)

def predict(text_list, config):
  model_path = config.get('path', "model_path")
  model = Doc2Vec.load(model_path)
  print(model.docvecs.most_similar([model.infer_vector(tokenize(text), epochs=50)]))

if __name__ == "__main__":
  config = ConfigUtil.get_instance().config
  text = get_text(config)
  text_list = tokenize(text)
  predict(text_list, config)