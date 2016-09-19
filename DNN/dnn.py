import tensorflow as tf
import numpy as np
import sys

from datahelper import DataHelper

VOCAB_SIZE=10000
EMBEDDING_SIZE=1
LEARNING_RATE=1e-3
MINI_BATCH_SIZE=256
NORMALIZE_LAYER=0

data_helper = DataHelper(_voc_size = VOCAB_SIZE)

data_helper.load_train_ins_and_process("data/train.50_51.ins")
data_helper.load_eval_ins("data/eval.52.ins")

print "data loaded"

def eval_auc(eval_res, eval_label):
    sorted_res = np.argsort(eval_res, axis=0)

    m = 0
    n = 0
    rank = 0

    for k in range(sorted_res.shape[0]):
        idx = sorted_res[k][0]
        if eval_label[idx][0] == 1:
            m += 1
            rank += k + 1
        else:
            n += 1

    sys.stdout.write("\nEval auc: %f\n" % ((rank * 1.0 - m * (m + 1) / 2.0) / (m * n * 1.0))  )
    sys.stdout.flush()
 

with tf.device("/cpu:0"):
    with tf.Session() as sess:

        x = []
        w_embedding = []

        epsilon = 1e-1
        decay = 0.999

        for slot in data_helper.slots:
            v_sz = len(data_helper.vocabulary[slot]["feat_dict"])

            x.append( tf.placeholder(tf.float32, shape = [None, v_sz + 1 ]) )
            w_embedding.append( tf.Variable( tf.truncated_normal([v_sz + 1, EMBEDDING_SIZE], stddev = 0.1) ) )

        y = tf.placeholder(tf.float32, shape=[None, 1])

        embedding_layer_sz = EMBEDDING_SIZE * data_helper.slot_size

        if NORMALIZE_LAYER == 1:
            embedding_scale = tf.Variable(tf.ones([embedding_layer_sz]))
            embedding_beta = tf.Variable(tf.zeros([embedding_layer_sz]))
            embedding_pop_mean = tf.Variable(tf.zeros([embedding_layer_sz]), trainable=False)
            embedding_pop_var = tf.Variable(tf.constant(0.1, shape=[embedding_layer_sz]),trainable=False)

        w_fc1 = tf.Variable(tf.truncated_normal([embedding_layer_sz, 1024], stddev = 0.1))
        b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))

        w_fc2 = tf.Variable(tf.truncated_normal([1024,512], stddev = 0.1))
        b_fc2 = tf.Variable(tf.constant(0.1,shape=[512]))

        w_fc3 = tf.Variable(tf.truncated_normal([512,256], stddev = 0.1))
        b_fc3 = tf.Variable(tf.constant(0.1,shape=[256]))

        w_fc4 = tf.Variable(tf.truncated_normal([256,128], stddev = 0.1))
        b_fc4 = tf.Variable(tf.constant(0.1,shape=[128]))

        w_fc5 = tf.Variable(tf.truncated_normal([128,64], stddev = 0.1))
        b_fc5 = tf.Variable(tf.constant(0.1,shape=[64]))

        w_fc6 = tf.Variable(tf.truncated_normal([64,1], stddev = 0.1))
        b_fc6 = tf.Variable(tf.constant(0.1,shape=[1]))

        #training network

        h_slot_embedding = []

        for i in range(len(w_embedding)):
            h_slot_embedding.append( tf.matmul(x[i], w_embedding[i]) )

        h_embedding = tf.concat(1, h_slot_embedding)
        
        if NORMALIZE_LAYER == 1:
            batch_mean, batch_var = tf.nn.moments(h_embedding,[0])
            train_mean = tf.assign(embedding_pop_mean, embedding_pop_mean * decay + batch_mean * (1-decay))
            train_var = tf.assign(embedding_pop_var, embedding_pop_var * decay + batch_var * (1-decay))

            with tf.control_dependencies([train_mean, train_var]):
                h_embedding_norm = tf.nn.batch_normalization(h_embedding, batch_mean, batch_var, embedding_beta, embedding_scale, epsilon)
        
            h_fc1 = tf.nn.relu(tf.matmul(h_embedding_norm, w_fc1) + b_fc1)
        else:
            h_fc1 = tf.nn.relu(tf.matmul(h_embedding, w_fc1) + b_fc1)

        h_fc2 = tf.matmul(h_fc1,w_fc2) + b_fc2
        h_fc3 = tf.matmul(h_fc2,w_fc3) + b_fc3
        h_fc4 = tf.matmul(h_fc3,w_fc4) + b_fc4
        h_fc5 = tf.matmul(h_fc4,w_fc5) + b_fc5

        output = tf.sigmoid(tf.matmul(h_fc5,w_fc6)+b_fc6)

        #eval network

        eval_x = tf.placeholder(tf.float32, shape=[None, data_helper.slot_size, EMBEDDING_SIZE])
        eval_x_merge = tf.reshape(eval_x, [-1, EMBEDDING_SIZE * data_helper.slot_size])
        
        if NORMALIZE_LAYER ==1 :
            eval_x_merge_norm = tf.nn.batch_normalization(eval_x_merge, embedding_pop_mean, embedding_pop_var, embedding_beta, embedding_scale, epsilon)
            eval_h_fc1 = tf.nn.relu(tf.matmul(eval_x_merge_norm, w_fc1) + b_fc1)
        else:
            eval_h_fc1 = tf.nn.relu(tf.matmul(eval_x_merge, w_fc1) + b_fc1)

        eval_h_fc2 = tf.matmul(eval_h_fc1, w_fc2) + b_fc2
        eval_h_fc3 = tf.matmul(eval_h_fc2, w_fc3) + b_fc3
        eval_h_fc4 = tf.matmul(eval_h_fc3, w_fc4) + b_fc4
        eval_h_fc5 = tf.matmul(eval_h_fc4, w_fc5) + b_fc5
        eval_output = tf.sigmoid(tf.matmul(eval_h_fc5, w_fc6) + b_fc6)

        with sess.as_default():

            loss = tf.reduce_mean(-( (1 - y)*tf.log(1 - output) + y * tf.log(output) ))
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            #optimizer = tf.train.AdagradOptimizer(5e-3)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_step = optimizer.apply_gradients(grads_and_vars)

            sess.run(tf.initialize_all_variables())

            epoch = 0
            while True:
                data_helper.shuffle_train_ins()
                
                sys.stdout.write("Epoch %d\n#################################################" % epoch)

                i = 0
                training_loss = 0
                
                training_ins_sz = len(data_helper.train_ins)

                offset = 0
                while offset < training_ins_sz:
                
                    if i % 50 == 0:
                        [ _w_embedding ] = sess.run([w_embedding])
                        eval_ins = data_helper.get_eval_ins_embedding(_w_embedding, EMBEDDING_SIZE)

                        eval_res = sess.run(eval_output, feed_dict = {eval_x:eval_ins})

                        eval_auc(eval_res, data_helper.eval_label)

                        sys.stdout.write("Mini batch trained:")
                        sys.stdout.flush()

                    [label, ins] = data_helper.get_next_batch(MINI_BATCH_SIZE, offset = offset)
                    offset += MINI_BATCH_SIZE


                    _feed_dict = { y : label }
                    for k in range(len(x)):
                        _feed_dict[x[k]] = ins[k]
                
                    if NORMALIZE_LAYER == 1:
                        _,_loss,_w_embedding, _scale, _beta, _pop_mean, _pop_var,_h_embedding, _h_embedding_norm, _h_fc1, _output \
                        = sess.run([train_step, loss, w_embedding, embedding_scale, embedding_beta, embedding_pop_mean, embedding_pop_var,h_embedding, h_embedding_norm, h_fc1, output], feed_dict=_feed_dict)
                    else:
                        _,_loss = sess.run([train_step, loss], feed_dict=_feed_dict)
                    
                    sys.stdout.write(" %d" % i)
                    sys.stdout.flush()
                    training_loss += _loss * len(ins)
                    
                    i += 1
                
                sys.stdout.write( "\n Epoch %d done: Training Loss: %f\n" % (epoch , training_loss / training_ins_sz ))
                
                sys.stdout.flush()

                epoch += 1

