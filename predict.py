#coding: UTF-8
import falcon
import json
import urllib.parse
from config_util import ConfigUtil
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer
class Resource(object):
  def on_get(self, req, resp):
    text = urllib.parse.unquote_to_bytes(req.params['input']).decode()
    result_json = main(text)
    resp.body = result_json
    resp.status = falcon.HTTP_200

def get_text(config):
  message = config.get('message','input_message')
  print(message)
  text = input('>>')
  return text # return like '僕の名前は'

def tokenize(text:str):
  tokenizer = Tokenizer()
  return tokenizer.tokenize(text, wakati=True) # return like ['僕の', '名前は']

def predict(text_list, config):
  model_path = config.get('path', "model_path")
  model = Doc2Vec.load(model_path)
  print(text_list)
  return model.docvecs.most_similar([model.infer_vector(text_list, epochs=50)]) #return like [('バラジャーノ', 0.83023), ('ほげ', 0.82)]

def tuple_list_to_json(result_tuple_list):
  result_json = json.dumps(result_tuple_list, ensure_ascii=False)
  return result_json # return like [["\u30d0\u30e9\u30b8\u30e3\u30fc\u30ce", 0.83023], ["\u307b\u3052", 0.82]] type: str

def main(text:str):
  config            = ConfigUtil.get_instance().config
  #text             =  get_text(config)
  #text              = 'お米ならいくらでも食べられます。'
  text_list         = tokenize(text)
  result_tuple_list = predict(text_list, config)
  #result_tuple_list = [('バラジャーノ', 0.83023), ('ほげ', 0.82)]
  result_json       = tuple_list_to_json(result_tuple_list)
  print(result_json)
  return result_json

""" if __name__ == "__main__":
  config = ConfigUtil.get_instance().config
  text = get_text(config)
  text_list = tokenize(text)
  print(predict(text_list, config)) """