from argparse import ArgumentParser
from keras.layers import LSTM, Embedding, Input, Concatenate, Masking, TimeDistributed, Dense, Bidirectional
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras_contrib.layers import CRF
import numpy as np


parser = ArgumentParser(description="Generate solution for CoNLL 2000 Sequence Chunking task using BiLSTM-CRF predictor")
parser.add_argument("--word-vector-features", action="store_true", help="use word vectors as features")
parser.add_argument("--crf-features", action="store_true", help="use standard crf features - only POS tag")
args = parser.parse_args()
if not args.word_vector_features and not args.crf_features:
    parser.error("At least one of --word-vector-features and --crf-features is required")


def file2features_ssi(filename):
    tokens = []
    pos_tags = []
    chunk_tags = []

    seq_split_indices = []

    for line in open(filename):
        s_line = line.rstrip().split(" ")
        if len(s_line) == 3:
            tokens.append(s_line[0])
            pos_tags.append(s_line[1])
            chunk_tags.append(s_line[2])
        else:
            seq_split_indices.append(len(tokens))

    return tokens, pos_tags, chunk_tags, seq_split_indices[:-2]


print("Loading train.txt...")
tokens, pos_tags, chunk_tags, ssi = file2features_ssi("train.txt")


print("Label encoding / one hot encoding training data, splitting training data into sequences and padding with zeros...")
X_inputs = []

if args.word_vector_features:
    token_tokenizer = Tokenizer(filters='\t\n', oov_token="oov_token")
    token_tokenizer.fit_on_texts(tokens)
    X_token = np.array(token_tokenizer.texts_to_sequences(tokens))

    # +1 for reserved index 0
    n_token_features = len(token_tokenizer.word_index) + 1

    X_token = np.squeeze(pad_sequences(np.split(X_token, ssi, axis=0), padding="post"))
    X_inputs.append(X_token)

if args.crf_features:
    pos_tags_tokenizer = Tokenizer(filters='\t\n')
    pos_tags_tokenizer.fit_on_texts(pos_tags)
    X_pos_tag = pos_tags_tokenizer.texts_to_matrix(pos_tags)

    n_pos_tag_features = len(pos_tags_tokenizer.word_index) + 1

    X_pos_tag = pad_sequences(np.split(X_pos_tag, ssi, axis=0), padding="post")
    X_inputs.append(X_pos_tag)

chunk_tags_tokenizer = Tokenizer(filters='\t\n')
chunk_tags_tokenizer.fit_on_texts(chunk_tags)
X_chunk_tag = chunk_tags_tokenizer.texts_to_matrix(chunk_tags)

n_chunk_tag_features = len(chunk_tags_tokenizer.word_index) + 1

X_chunk_tag = pad_sequences(np.split(X_chunk_tag, ssi, axis=0), padding="post")


def load_glove_embeddings_index(filename):
    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def calc_embedding_matrix(embeddings_index, embedding_dim, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


if args.word_vector_features:
    print("Loading glove.6B.100d.txt...")
    embeddings_index = load_glove_embeddings_index("glove.6B.100d.txt")

    print("Calculating glove embedding matrix...")
    embedding_dim = 100
    embedding_matrix = calc_embedding_matrix(embeddings_index, embedding_dim, token_tokenizer.word_index)


print("Compiling LSTM model...")
num_seqs = X_chunk_tag.shape[0]
max_seq_size = X_chunk_tag.shape[1]


def compile_model(batch_size):
    if args.word_vector_features:
        token_input = Input(batch_shape=(batch_size, max_seq_size))
        token_emb = Embedding(
            n_token_features,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_size,
            trainable=False,
            mask_zero=True)(token_input)

    if args.crf_features:
        pos_tag_input = Input(batch_shape=(batch_size, max_seq_size, n_pos_tag_features))

    if args.word_vector_features and args.crf_features:
        inputs = [token_input, pos_tag_input]
        concat_output = Concatenate()([token_emb, pos_tag_input])
        mask_output = Masking()(concat_output)
    elif args.word_vector_features:
        inputs = [token_input]
        mask_output = token_emb
    else:
        inputs = [pos_tag_input]
        mask_output = Masking()(pos_tag_input)

    lstm1_output = Bidirectional(LSTM(
        100,
        stateful=False,
        return_sequences=True,
        dropout=0.1,
        recurrent_dropout=0.1))(mask_output)
    lstm2_output = Bidirectional(LSTM(
        100,
        stateful=False,
        return_sequences=True,
        dropout=0.1,
        recurrent_dropout=0.1))(lstm1_output)
    time_dist_output = TimeDistributed(Dense(n_chunk_tag_features))(lstm2_output)
    crf = CRF(n_chunk_tag_features)
    chunk_tag_output = crf(time_dist_output)

    model = Model(inputs=inputs, outputs=[chunk_tag_output])
    optimizer=Nadam(clipnorm=1.)
    model.compile(optimizer=optimizer, loss=crf.loss_function)
    return model


train_batch_size = 5
model = compile_model(train_batch_size)
print(model.summary())


print("Training LSTM model...")
model.fit(X_inputs, X_chunk_tag, batch_size=train_batch_size, epochs=25, shuffle=True)


print("Loading and one hot encoding test data...")
tokens, pos_tags, chunk_tags, ssi = file2features_ssi("test.txt")


print("Label encoding / one hot encoding test data, splitting test data into sequences and padding with zeros...")
X_inputs = []

if args.word_vector_features:
    X_token = np.array([t if t else [0] for t in token_tokenizer.texts_to_sequences(tokens)])
    X_token = np.squeeze(pad_sequences(np.split(X_token, ssi, axis=0), padding="post", maxlen=max_seq_size))
    X_inputs.append(X_token)

if args.crf_features:
    X_pos_tag = pos_tags_tokenizer.texts_to_matrix(pos_tags)
    X_pos_tag = pad_sequences(np.split(X_pos_tag, ssi, axis=0), padding="post", maxlen=max_seq_size)
    X_inputs.append(X_pos_tag)

X_chunk_tag = chunk_tags_tokenizer.texts_to_matrix(chunk_tags)
X_chunk_tag = pad_sequences(np.split(X_chunk_tag, ssi, axis=0), padding="post", maxlen=max_seq_size)


print("Making predictions for test data...")
test_batch_size = 1
new_model = compile_model(test_batch_size)
new_model.set_weights(model.get_weights())

X_chunk_tag_prob = new_model.predict(X_inputs, batch_size=test_batch_size)
X_chunk_tag_pred = X_chunk_tag_prob.argmax(axis=-1)


def reverse_word_index(word_index):
    index_word = {v: k for k, v in word_index.items()}
    index_word[0] = ""
    return index_word


def reverse_encoding(X, word_index):
    index_word = reverse_word_index(word_index)
    return [[index_word[idx].upper() for idx in seq] for seq in X]


R_chunk_tag_pred = reverse_encoding(X_chunk_tag_pred, chunk_tags_tokenizer.word_index)


with open("test.txt") as test_file:
    with open("conlleval_input_bilstmcrf.txt", "w") as conlleval_input_file:
        R_chunk_tag_pred_iter = iter(R_chunk_tag_pred)
        R_chunk_tag_pred_iter_iter = iter(next(R_chunk_tag_pred_iter))
        for line in test_file:
            if line == "\n":
                conlleval_input_file.write(line)
                R_chunk_tag_pred_iter_iter = iter(next(R_chunk_tag_pred_iter))
            else:
                line = line.rstrip()
                conlleval_input_file.write("{} {}\n".format(line, next(R_chunk_tag_pred_iter_iter)))

