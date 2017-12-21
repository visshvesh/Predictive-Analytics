from grpc.beta import implementations
import numpy
import tensorflow as tf
from datetime import datetime
#from tensorflow_serving_client import  predict_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('errorcode',150, 'path to errorcode')
FLAGS = tf.app.flags.FLAGS
#def do_inference(hostport):
def do_inference(hostport):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the Prediction Service.
    Returns:
    pred values, ground truth label
    """
    # create connection
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # initialize a request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'saved_model'
    request.model_spec.signature_name = 'prediction'
    data = FLAGS.errorcode

    # Randomly generate some test data
    # temp_data = numpy.random.randn(10, 3).astype(numpy.float32)
    # data, label = temp_data, numpy.sum(temp_data * numpy.array([1, 2, 3]).astype(numpy.float32), 1)
    # request.inputs['input'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(data, shape=data.shape))



    # # Randomly generate some test data
    # temp_data = numpy.random.randn(10, 3).astype(numpy.float32)
    # data, label = temp_data, numpy.sum(temp_data * numpy.array([1, 2, 3]).astype(numpy.float32), 1)
    request.inputs['input'].CopyFrom(
         tf.contrib.util.make_tensor_proto(data))

    # predict
    result = stub.Predict(request, 30.0)  # 5 seconds
    return result


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    result = do_inference(FLAGS.server)
    print('Result is: ', result)
    #print('Actual label is: ', label)


if __name__ == '__main__':
    tf.app.run()