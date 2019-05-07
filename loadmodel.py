import tensorflow as tf
import preproc

sess = tf.Session()
# 加载图和参数变量
saver = tf.train.import_meta_graph('./ckpt/model20190505_160937.ckpt.meta')
graph = tf.get_default_graph()
b = graph.get_tensor_by_name("w:0")
# saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))

# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = preproc.datapre()