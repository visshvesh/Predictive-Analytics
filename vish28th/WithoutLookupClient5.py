## Import Library
import tensorflow as tf

## Within Session
with tf.Session(graph=tf.Graph()) as sess:
    ## Load model
    export_dir='./1'
    model=tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],export_dir)
    loaded_graph=tf.get_default_graph()
    ## Get Input and output from model
    input_tensor_name=model.signature_def['prediction'].inputs['input'].name
    input_tensor=loaded_graph.get_tensor_by_name(input_tensor_name)
    output_tensor_name=model.signature_def['prediction'].outputs['output'].name
    output_tensor=loaded_graph.get_tensor_by_name(output_tensor_name)
    ## Get the result
    result=sess.run(output_tensor,{input_tensor:"169"})
    print(result)