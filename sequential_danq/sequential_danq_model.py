import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

tf.reset_default_graph()

def getSequentialDanQ(batchSize=64, convWindow=26, convFilters=320,
                    poolWindow=13, lstmTimeSteps=51, lstmSize=320,
                    contextVectorSize=160):
    """ (Default) architecture:
        CONV1D-64x320 ReLU
        POOL1D-13
        Dropout
        BLSTM-320-320
        Dropout
        FC-160 ReLU
        DecoderLSTM-160
        FC-1
    """

    seqLength = lstmTimeSteps * poolWindow

    x = tf.placeholder(tf.float32, [batchSize, seqLength, 4], name="x")
    x_rshp = tf.reshape(x, [batchSize, seqLength, 1, 4], name="x_rshp")
    # Shape = [batchSize, seqLength, 1, 4]

    ### Convolutional layer
    W_conv = tf.Variable(tf.truncated_normal([convWindow, 1, 4, convFilters],
                                             stddev=0.1), name="W_conv")
    b_conv = tf.Variable(tf.constant(0.1, shape=[convFilters]), name="b_conv")
    h_conv = tf.nn.relu(tf.nn.conv2d(x_rshp, W_conv, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv, name="h_conv")
    # Shape = [batchSize, seqLength, 1, convFilters]

    ### 1D max-pooling layer
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, poolWindow, 1, 1],
                            strides=[1, poolWindow, 1, 1], padding='SAME',
                            name="h_pool")
    # Shape = [batchSize, lstmTimeSteps, 1, convFilters]

    ### Dropout
    keep_prob_pool = tf.placeholder(tf.float32, name="keep_prob_pool")
    h_pool_drop = tf.nn.dropout(h_pool, keep_prob_pool, name="h_pool_drop")

    h_pool_drop_rshp = tf.reshape(h_pool_drop, [-1, lstmTimeSteps, convFilters],
                                  name="h_pool_drop_rshp")
    # Shape = [batchSize, lstmTimeSteps, convFilters]

    ### Encoder: Bidirectional LSTM
    lstm_in_list = tf.unpack(h_pool_drop_rshp, axis=1)

    with tf.variable_scope('forward'):
        lstm_fw = rnn_cell.BasicLSTMCell(lstmSize, forget_bias=1.0)
    with tf.variable_scope('backward'):
        lstm_bw = rnn_cell.BasicLSTMCell(lstmSize, forget_bias=1.0)
    with tf.variable_scope('blstm', reuse=None):
        outputs, state_fw, state_bw = rnn.bidirectional_rnn(
                lstm_fw, lstm_bw, lstm_in_list, dtype="float")

    # outputs shape = [batchSize, lstmTimeSteps, 2 * lstmSize (fw + bw)]

    h_lstm = tf.reshape(outputs, [-1, lstmTimeSteps * 2 * lstmSize])

    ### Encoder: Dropout
    keep_prob_lstm = tf.placeholder(tf.float32, name="keep_prob_lstm")
    h_lstm_drop = tf.nn.dropout(h_lstm, keep_prob_lstm)
    # Shape = [batchSize, lstmTimeSteps * 2 * lstmSize]

    ### Encoder: Fully-connected layer
    W_blstm = tf.Variable(tf.truncated_normal(
        [lstmTimeSteps * 2 * lstmSize, contextVectorSize], stddev=0.1,
        name="W_blstm"))
    b_blstm = tf.Variable(tf.constant(0.1, shape=[contextVectorSize]),
            name="b_blstm")

    decoder_output = tf.nn.relu(tf.matmul(h_lstm_drop, W_blstm) + b_blstm,
            name="decoder_output")
    # Shape = [batchSize, contextVectorSize]

    with tf.variable_scope('decode') as scope:
        ### Decoder: LSTM
        decoder_lstm = rnn_cell.BasicLSTMCell(contextVectorSize)
        state = decoder_lstm.zero_state(batchSize, tf.float32)
        y_list = list()

        ### Decoder: Fully-connected layer
        W_decode = tf.Variable(tf.truncated_normal([contextVectorSize, 1],
            stddev=0.1, name="W_decode"))
        b_decode = tf.Variable(tf.constant(0.1, shape=[1]), name="b_decode")

        for i in range(seqLength):
            if i > 0:
                scope.reuse_variables()
            decoder_output, state = decoder_lstm(decoder_output, state)
            y_list.append(tf.matmul(decoder_output, W_decode) + b_decode)
    # y_list shape: list of seqLength [batchSize, 1]

    y = tf.pack(y_list, axis=1, name="y")
    # Shape = [batchSize, seqLength, 1]

    y_ = tf.placeholder(tf.float32, [batchSize, seqLength, 1], name="y_")
    return x, keep_prob_pool, keep_prob_lstm, y, y_
