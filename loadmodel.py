import tensorflow as tf
import preproc

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, "ckpt/model20190609_190518.ckpt")

    coord = tf.train.Coordinator()