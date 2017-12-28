import pandas as pd
import tensorflow as tf
import numpy as np
import os
data = pd.read_csv("MapData.csv", encoding="ISO-8859-1", usecols=["ERROR_CODE", "Criteria"])
ls = data.ERROR_CODE.tolist()
Criteria = data.Criteria.tolist()

for k in range(len(ls)):
    ls[k] = str(ls[k])

for k in range(len(Criteria)):
    Criteria[k] = str(Criteria[k])


tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '', 'Working directory.')
tf.app.flags.DEFINE_string('errorcode', '', 'path to errorcode')
FLAGS = tf.app.flags.FLAGS


keys = tf.constant(ls)
keys_var = tf.Variable(keys, name="keys_var")
values = tf.constant(Criteria)
value_var = tf.Variable(values, name="value_var")
saver = tf.train.Saver()

x = tf.placeholder(tf.string)
feed_dict ={x:'150'}

def main(_):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        k = keys_var.eval()
        v = value_var.eval()
        d = x.eval(feed_dict=feed_dict)
        # input_tensor = tf.constant(["100"])
        input_tensor = tf.constant(d)
        # x1 = tf.as_string(input_tensor)
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(k, v), "Unknown Error Code")
        out = table.lookup(input_tensor)
        table.init.run()

        print(d)
        output = out.eval()
        print(output)
        # save_path = saver.save(sess, "model_table1/model.ckpt")
        # print("Model saved in file: %s" % save_path)


        export_path_base = FLAGS.work_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tensor)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(out)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': tensor_info_x},
                outputs={'output': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'prediction':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()