# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import data_helpers
from tensorflow.contrib import learn
import os

class TextRNN:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer
        self.num_sampled=20


        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        #print(self.input_y.shape)

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        self.loss_val = self.loss() #-->self.loss_nce()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        #2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32) #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        print("outputs:===>",outputs) #outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        #3. concat output
        output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        self.output_rnn_last=tf.reduce_mean(output_rnn,axis=1) #[batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
        print("output_rnn_last:", self.output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        #4. logits(use linear layer)
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def loss_nce(self,l2_lambda=0.0001): #0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training: #training
            #labels=tf.reshape(self.input_y,[-1])               #[batch_size,1]------>[batch_size,]
            labels=tf.expand_dims(self.input_y,1)                   #[batch_size,]----->[batch_size,1]
            loss = tf.reduce_mean( #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),#[hidden_size*2, num_classes]--->[num_classes,hidden_size*2]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                               biases=self.b_projection,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                               labels=labels,                 #[batch_size,1]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
                               inputs=self.output_rnn_last,# [batch_size,hidden_size*2] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                               num_sampled=self.num_sampled,  #scalar. 100
                               num_classes=self.num_classes,partition_strategy="div"))  #scalar. 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.

    tf.flags.DEFINE_string("feature_file", "data/sentence.txt", "feature data (sentence).")
    tf.flags.DEFINE_string("label_file", "data/label.txt", "label data (number).")
    tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.feature_file, FLAGS.label_file)

    y = np.array(y)

    outsize = 6
    new_ydata = []
    for i in range(len(y)):
      new_ydata.append([0]*outsize)
      new_ydata[i][y[i]] = 1   
      #print(new_ydata[i])
    new_ydata = np.array(new_ydata)
    new_y = new_ydata

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))

    x_shuffled = np.array(x)[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    num_classes=6
    learning_rate=0.0001
    batch_size=x_train.shape[0]
    decay_steps=1000
    decay_rate=0.9
    sequence_length=max_document_length
    vocab_size=20570
    embed_size=100
    is_training=True
    dropout_keep_prob=0.5
    textRNN=TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = tf.ConfigProto(allow_soft_placement=True)          #my modification
    config.gpu_options.allow_growth = True

    saver=tf.train.Saver()
    is_train = False

    with tf.Graph().device('/gpu:3'), tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if is_train:
            for kk in range(30):
                # Generate batches
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                step = 0
                # Training loop. For each batch...
                tmp_batches = batches
                for batch in tmp_batches:
                    x_batch, y_batch = zip(*batch)
                    #train_step(x_batch, y_batch)
                    input_x = x_batch
                    input_y = y_batch
                    loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
                    print("iteration", kk, "step", step, "loss:",loss,"acc:",acc)#"label:",input_y,"prediction:",predict)
                    step += 1
                    if step%100 == 0:
                        loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:x_dev,textRNN.input_y:y_dev,textRNN.dropout_keep_prob:dropout_keep_prob})
                        print("**********************************", "iteration", kk, "step", step, "loss:",loss,"acc:",acc)#"label:",input_y,"prediction:",predict)

                        saver=tf.train.Saver(max_to_keep=1)
                        saver.save(sess,'rnnmodel/net.ckpt')
        else:
            saver.restore(sess, 'rnnmodel/net.ckpt')
            print("reload success!")
            loss, acc, predict,_=sess.run([textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.train_op],feed_dict={textRNN.input_x:x_dev,textRNN.input_y:y_dev,textRNN.dropout_keep_prob:1})
            print("\n\nmodel accuracy on test data is: {}%\n\n".format(acc*100))

test()
