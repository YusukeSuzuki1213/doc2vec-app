import falcon
import predict

api = application = falcon.API()
predict = predict.Resource()
api.add_route('/predict', predict)