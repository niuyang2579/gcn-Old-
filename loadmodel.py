import tensorflow as tf

sess = tf.Session()
# 加载图和参数变量
saver = tf.train.import_meta_graph('./ckpt/model20190505_155408.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))