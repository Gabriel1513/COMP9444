import tensorflow as tf
import re
import string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 145  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """


    rmbr = re.sub(r'<br />', ' ', review)
    rmpunc = ''.join(c for c in rmbr if c not in string.punctuation)
    rmpunc = rmpunc.lower()
    words = rmpunc.split(" ")
    processed_review = []
    for word in words:
        if word not in stop_words:
            processed_review.append(word)
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss

    """
    lstmUnits = 64
    numClasses = 2
    
    input_data = tf.placeholder(tf.float32, shape = [BATCH_SIZE, MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE], name = "input_data")
    labels = tf.placeholder(tf.int32, shape = (BATCH_SIZE, 2), name = "labels")
    dropout_keep_prob = tf.placeholder_with_default(0.75, shape = (), name = "dropout_keep_prob")
    
    lstmCell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)

    value, _ = tf.nn.dynamic_rnn(cell=lstmCell, inputs=input_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits,numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

    value = tf.transpose(value, [1,0,2])
    last = tf.gather(value, int(value.get_shape()[0])-1)
    prediction = (tf.matmul(last,weight)+bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32),name = 'accuracy')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = labels), name= 'loss')
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
