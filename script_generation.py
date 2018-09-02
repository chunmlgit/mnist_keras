import helper
import numpy as np
import problem_unittests as tests
from collections import Counter
import tensorflow as tf

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
text = text[81:]

view_sentence_range = (0, 10)
print('Dataset stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))

sentence_count_per_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_per_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))

word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

def create_lookup_tables(text):
    word_count = Counter(text)
    vocab_to_int = {word:idx for idx, word in enumerate(word_count)}
    int_to_vocab = {idx:word for idx, word in enumerate(word_count)}

    return vocab_to_int, int_to_vocab

tests.test_create_lookup_tables(create_lookup_tables)

def token_lookup():
    return {'.': '||Period||',
            ',': '||Comma||',
            '?': '||Question_Mark||',
            '"': '||Quotation_Mark||',
            ';': '||Semicolon||',
            '!': '||Exclamation_Mark||',
            '(': '||Left_Parenthese||',
            ')': '||Right_Parenthese||',
            '--': '||Dash||',
            '\n': '||Return||'}

tests.test_tokenize(token_lookup)

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs, targets, learning_rate

tests.test_get_inputs(get_inputs)

def get_init_cell(batch_size, lstm_size):
    num_layers = 2
    def build_cell(rnn_size):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        return lstm

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size) for _ in range(num_layers)])
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')

    return cell, initial_state

tests.test_get_init_cell(get_init_cell)

def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)

    return embed

tests.test_get_embed(get_embed)

def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')

    return outputs, final_state

tests.test_build_rnn(build_rnn)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    embed_mat = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed_mat)
    outputs_shape = outputs.get_shape()
    outputs = tf.concat(outputs, axis=1)
    outputs = tf.reshape(outputs, [-1, rnn_size])

    softmax_w = tf.Variable(tf.truncated_normal((rnn_size, vocab_size), stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    logits = tf.matmul(outputs, softmax_w) + softmax_b
    logits = tf.reshape(logits, [outputs_shape[0], outputs_shape[1], vocab_size])

    return logits, final_state

tests.test_build_nn(build_nn)

def get_batches(int_text, batch_size, seq_length):
    num_of_words = batch_size * seq_length
    n_batches = len(int_text) // num_of_words
    int_text = int_text[:n_batches * num_of_words]
    int_text_temp = int_text.copy()
    int_text_target = np.zeros_like(int_text)
    int_text_target[:-1] = int_text_temp[1:]
    int_text_target[-1] = int_text_temp[0]
    int_text = np.reshape(int_text, (n_batches, -1))
    int_text_target = np.reshape(int_text_target, (n_batches, -1))

    batches = np.zeros((n_batches, 2, batch_size, seq_length))
    #for i in range(0, n_batches):
    for j in range(0, batch_size):
        batches[:, 0, j, :] = int_text[i, :]
        batches[i, 1, :, :] = int_text_target[i, :].reshape((batch_size, seq_length))
    return batches

#tests.test_get_batches(get_batches)
