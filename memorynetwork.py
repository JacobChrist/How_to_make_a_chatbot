'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''
from __future__ import print_function

from keras.models import Sequential, Model, model_from_json
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import json


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def startup():
  try:
      path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
  except:
      print('Error downloading dataset, please download it manually:\n'
            '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
            '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
      raise
    
  tar = tarfile.open(path)

  challenges = {
      # QA1 with 10,000 samples
      'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
      # QA2 with 10,000 samples
      'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
  }
  challenge_type = 'single_supporting_fact_10k'
  challenge = challenges[challenge_type]

  print('Extracting stories for the challenge:', challenge_type)
  train_stories = get_stories(tar.extractfile(challenge.format('train')))
  test_stories = get_stories(tar.extractfile(challenge.format('test')))
  print(test_stories)
  
  vocab = set()
  for story, q, answer in train_stories + test_stories:
      vocab |= set(story + q + [answer])
  vocab = sorted(vocab)
      
  # Reserve 0 for masking via pad_sequences
  vocab_size = len(vocab) + 1
  story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
  query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

  print('-')
  print('Vocab size:', vocab_size, 'unique words')
  print('Story max length:', story_maxlen, 'words')
  print('Query max length:', query_maxlen, 'words')
  print('Number of training stories:', len(train_stories))
  print('Number of test stories:', len(test_stories))
  print('-')
  print('Here\'s what a "story" tuple looks like (input, query, answer):')
  print(train_stories[0])
  #for t in range(0, len(train_stories)):
  #  print(t, train_stories[t])
  print('-')
  print('Vectorizing the word sequences...')

  word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
  idx_word = dict((i + 1, c) for i, c in enumerate(vocab))
  inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                                 word_idx,
                                                                 story_maxlen,
                                                                 query_maxlen)
                                                                 
  print(inputs_train[0])
  print(queries_train[0])
  print(answers_train[0])
  print(vocab)

  inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                              word_idx,
                                                              story_maxlen,
                                                              query_maxlen)

  print('-')
  print('inputs: integer tensor of shape (samples, max_length)')
  print('inputs_train shape:', inputs_train.shape)
  print('inputs_test shape:', inputs_test.shape)
  print('-')
  print('queries: integer tensor of shape (samples, max_length)')
  print('queries_train shape:', queries_train.shape)
  print('queries_test shape:', queries_test.shape)
  print('-')
  print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
  print('answers_train shape:', answers_train.shape)
  print('answers_test shape:', answers_test.shape)
  print('-')

  return inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, train_stories, test_stories, story_maxlen, query_maxlen, vocab_size
  #print('Exiting...')
  #exit()

def compile(inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, train_stories, test_stories, story_maxlen, query_maxlen, vocab_size):
  print('Compiling...')
  # placeholders
  input_sequence = Input((story_maxlen,))
  question = Input((query_maxlen,))

  # encoders
  # embed the input sequence into a sequence of vectors
  input_encoder_m = Sequential()
  input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
  input_encoder_m.add(Dropout(0.3))
  # output: (samples, story_maxlen, embedding_dim)

  # embed the input into a sequence of vectors of size query_maxlen
  input_encoder_c = Sequential()
  input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
  input_encoder_c.add(Dropout(0.3))
  # output: (samples, story_maxlen, query_maxlen)

  # embed the question into a sequence of vectors
  question_encoder = Sequential()
  question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
  question_encoder.add(Dropout(0.3))
  # output: (samples, query_maxlen, embedding_dim)

  # encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  input_encoded_m = input_encoder_m(input_sequence)
  input_encoded_c = input_encoder_c(input_sequence)
  question_encoded = question_encoder(question)

  # compute a 'match' between the first input vector sequence
  # and the question vector sequence
  # shape: `(samples, story_maxlen, query_maxlen)`
  match = dot([input_encoded_m, question_encoded], axes=(2, 2))
  match = Activation('softmax')(match)

  # add the match matrix with the second input vector sequence
  response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
  response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

  # concatenate the match matrix with the question vector sequence
  answer = concatenate([response, question_encoded])

  # the original paper uses a matrix multiplication for this reduction step.
  # we choose to use a RNN instead.
  answer = LSTM(32)(answer)  # (samples, 32)

  # one regularization layer -- more would probably be needed.
  answer = Dropout(0.3)(answer)
  answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
  # we output a probability distribution over the vocabulary
  answer = Activation('softmax')(answer)

  # build the final model
  model = Model([input_sequence, question], answer)
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

  # train
  model.fit([inputs_train, queries_train], answers_train, batch_size=32, epochs=120,
            validation_data=([inputs_test, queries_test], answers_test))


  #write out the vocab as json file
  with open('vocab.json', 'w') as outfile:
    json.dump(vocab, outfile)
    
  #yaml_string = model.to_yaml()
  #with open("model.yaml", "w") as text_file:
  #    text_file.write(yaml_string)

  json_string = model.to_json()
  with open("model.json", "w") as text_file:
    text_file.write(json_string)

  model.save_weights('weights.hdf5')  
  
def trained(inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, train_stories, test_stories, story_maxlen, query_maxlen, vocab_size):
  print("****************************")
  print("******* Not training *******")
  print("****************************")
  
  #print("YAML Model:")
  #with open ("model.yaml", "r") as myfile:
  #  yaml_string=myfile.read()
  #print(yaml_string)


  print("Importing JSON vocab:")
  with open('vocab.json') as data_file:    
      vocab = json.load(data_file)
  word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
  idx_word = dict((i + 1, c) for i, c in enumerate(vocab))
  #print(vocab)
  #print(word_idx)
  #print(idx_word)

  print("Importing JSON Model:")
  with open ("model.json", "r") as myfile:
    json_string=myfile.read()
  #print(json_string)
  model = model_from_json(json_string)

  print("Importing Model Weights:")
  model.load_weights('weights.hdf5')


  print("Prediction:")
  #9995 (['John', 'moved', 'to', 'the', 'bathroom', '.', 'Daniel', 'went', 'to', 'the', 'kitchen', '.'], ['Where', 'is', 'John', '?'], 'bathroom')
  #9996 (['John', 'moved', 'to', 'the', 'bathroom', '.', 'Daniel', 'went', 'to', 'the', 'kitchen', '.', 'Sandra', 'travelled', 'to', 'the', 'kitchen', '.', 'Mary', 'travelled', 'to', 'the', 'hallway', '.'], ['Where', 'is', 'Daniel', '?'], 'kitchen')
  #9997 (['John', 'moved', 'to', 'the', 'bathroom', '.', 'Daniel', 'went', 'to', 'the', 'kitchen', '.', 'Sandra', 'travelled', 'to', 'the', 'kitchen', '.', 'Mary', 'travelled', 'to', 'the', 'hallway', '.', 'Sandra', 'went', 'back', 'to', 'the', 'bathroom', '.', 'John', 'went', 'back', 'to', 'the', 'kitchen', '.'], ['Where', 'is', 'Mary', '?'], 'hallway')
  #9998 (['John', 'moved', 'to', 'the', 'bathroom', '.', 'Daniel', 'went', 'to', 'the', 'kitchen', '.', 'Sandra', 'travelled', 'to', 'the', 'kitchen', '.', 'Mary', 'travelled', 'to', 'the', 'hallway', '.', 'Sandra', 'went', 'back', 'to', 'the', 'bathroom', '.', 'John', 'went', 'back', 'to', 'the', 'kitchen', '.', 'Daniel', 'went', 'back', 'to', 'the', 'office', '.', 'Daniel', 'journeyed', 'to', 'the', 'bathroom', '.'], ['Where', 'is', 'John', '?'], 'kitchen')
  #9999 (['John', 'moved', 'to', 'the', 'bathroom', '.', 'Daniel', 'went', 'to', 'the', 'kitchen', '.', 'Sandra', 'travelled', 'to', 'the', 'kitchen', '.', 'Mary', 'travelled', 'to', 'the', 'hallway', '.', 'Sandra', 'went', 'back', 'to', 'the', 'bathroom', '.', 'John', 'went', 'back', 'to', 'the', 'kitchen', '.', 'Daniel', 'went', 'back', 'to', 'the', 'office', '.', 'Daniel', 'journeyed', 'to', 'the', 'bathroom', '.', 'John', 'went', 'back', 'to', 'the', 'office', '.', 'Mary', 'travelled', 'to', 'the', 'bedroom', '.'], ['Where', 'is', 'John', '?'], 'office')
  #question = np.array([['John', 'moved', 'to', 'the', 'bathroom', '.', 'Daniel', 'went', 'to', 'the', 'kitchen', '.'], ['Where', 'is', 'John', '?']])
  question = [inputs_test, queries_test]
  #print(np.shape(question))
  print(np.shape(inputs_test))
  print(np.shape(queries_test))
  print(question)
  answer = model.predict(question)
  print(np.rint(answer[0]))
  print(test_stories[0])

  user_question = "go"
  while user_question != "quit":
    user_question = input("Ask a question or quit: ")
    print("You asked: " + user_question)
    num=user_question.count(" ")
    query = user_question.split(' ', num )
    print("Who do I look like your slavebot? Read the damn story yourself!")
    #query = np.array(user_question.split(' ', num ))
    #print(query)
    #print(user_question.split(' ', num ))
    
    #inputs_test, queries_test, answers_test = vectorize_stories([[]],word_idx,story_maxlen,query_maxlen)


#####################################################################################################
# chatbot
#####################################################################################################
# Take a look at how the training data is stored (incase I want to add some additional data)
# Figure out how to save Keras Model (and weights) so that I don't have to train it over and over.
# First two steps were easy enough, decided to try to do some prediction.
# Keras prediction function looks easy enough, but I probably need to ask the question the right way
# -> This is causing me to read code to try an understand how to ask the question
# -> If I can get and answer I am concerned that the answer I get may be some vectors
# --> I'm assuming that if the answer is in vector form that I will need to decode back to english
# Stored vectors integers from the vocab lookup table, this table will need to be saved and loaded.
          
if __name__ == '__main__':
  print("main")  
  inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, train_stories, test_stories, story_maxlen, query_maxlen, vocab_size = startup()
  print(inputs_test)
  print(queries_test)
  train = input("Train (yes/no): ")
  if train == "yes" or train == "y":
    compile(inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, train_stories, test_stories, story_maxlen, query_maxlen, vocab_size)
  trained(inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, train_stories, test_stories, story_maxlen, query_maxlen, vocab_size)
  exit()

