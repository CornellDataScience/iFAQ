from flask import Flask, flash, request, render_template
import torch, pickle
import torch.nn as nn
import scipy.optimize
import numpy as np
import lstm_utils as u
import lstm_constants as c
import re
from nltk.stem.wordnet import WordNetLemmatizer
# also run nltk.download('wordnet')
# How can I find God? -> How do I find God?


from candidate_retrieval import CandidateStore
from predictor import Predictor
from multiprocessing import cpu_count

from collections import OrderedDict

class LstmNet(nn.Module):

    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size = c.SENT_INCLUSION_MAX,
                            hidden_size = 200,
                            num_layers = 2)
        self.fc1 = nn.Linear(200,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50,2)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[-1]
        x = nn.LeakyReLU()(self.bn1(self.fc1(x)))
        x = nn.LeakyReLU()(self.bn2(self.fc2(x)))
        x = nn.Softmax(dim=1)(self.fc3(x))
        return x

def lemmatizer(word):
    """Returns: lemmatized word if word >= length 5
    """
    if len(word)<4:
        return word
    return wnl.lemmatize(wnl.lemmatize(word, "n"), "v")

def clean_string(string): # From kaggle-quora-dup submission
    """Returns: cleaned string, with common token replacements and lemmatization
    """
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    string = ' '.join([lemmatizer(w) for w in string.split()])
    return string

def get_matrix(str1, str2):
    goal_str = str1; use_str = str2
    goal_vecs = u.vector_list(word2vect, goal_str)
    use_vecs = np.array(u.vector_list(word2vect, use_str))
    matrix = np.zeros((c.SENT_INCLUSION_MAX,c.SENT_INCLUSION_MAX))

    for g_idx in range(len(goal_vecs)):
        goal_vec = goal_vecs[g_idx]
        if (goal_vec == np.zeros((c.WORD_EMBED_DIM))).all():
            matrix[g_idx] = np.zeros((c.SENT_INCLUSION_MAX))
        else:
            objective = lambda weights: u.custom_entropy(np.array(weights))
            init_guess = [0.0001]*c.SENT_INCLUSION_MAX
            cons_func1 = lambda weights: np.array(weights).dot(use_vecs)-goal_vec+0.001
            cons_func2 = lambda weights: -np.array(weights).dot(use_vecs)+goal_vec+0.001
            constraint1 = {'type':'ineq','fun':cons_func1}
            constraint2 = {'type':'ineq','fun':cons_func2}
            bound = scipy.optimize.Bounds(0.,1.)
            res = scipy.optimize.minimize(objective, init_guess, 
                                        method='SLSQP', 
                                        constraints=[constraint1,constraint2],
                                        bounds=bound)
            matrix[g_idx] = np.nan_to_num(res.x)
    return matrix

def duplicate_in_cache(question):
    for key in cache:
        q1, q2 = clean_string(question), clean_string(key) 
        print('Comparing:\t{}\t{}'.format(q1,q2))
        matrix = np.expand_dims(get_matrix(q1, q2), 0)
        matrix = np.swapaxes(matrix, 0, 1)
        matrix = torch.tensor(matrix, dtype=torch.float32)
        pred = int(lstm_model(matrix).max(1)[1])
        print('Prediction:{}'.format(pred))
        if pred==1:
            print('pred is 1!')
            return True, cache[key]
    return False, None


app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question = request.form['question']
        if len(question) == 0:
            answer = ''
        elif question in cache:
            answer = cache[question]
        else:
            is_dup_in, dup_ans = duplicate_in_cache(question)
            if is_dup_in:
                print('Detected non-identical duplicate! Returning answer...')
                answer = dup_ans
            else:
                candidate_paragraphs = cs.retrieve(question)
                candidate_answers = []
                for context in candidate_paragraphs.values:
                    print(context)
                    prediction = model.predict(context, question, None, 1)
                    candidate_answers.append(prediction[0])

                candidate_answers.sort(key=lambda x: x[1], reverse=True)
                answer = candidate_answers[0][0]
                if len(cache) >= cache_capacity:
                    cache.popitem(last=False)
                cache[question] = answer

        return render_template('home.html', question=question, answer=answer)


@app.route('/documents')
def documents():
    return render_template('documents.html', files=cs.file_info)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('documents.html', files=cs.file_info)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.')
            return render_template('documents.html', files=cs.file_info)

        if not file.filename.endswith('txt'):
            flash('Invalid file format')
            return render_template('documents.html', files=cs.file_info)

        # Add file to candidate store
        cs.add_doc(file.filename)

    return render_template('documents.html', files=cs.file_info)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    # load ml model
    model = Predictor(
        'app/static/m_reader.mdl',
        normalize=True,
        embedding_file=None,
        char_embedding_file=None,
        num_workers=cpu_count() // 2
    )

    print('Loading LSTM...')
    lstm_model = torch.load('app/static/better_net.pt',map_location=torch.device('cpu'))
    lstm_model.eval()

    print('Loading GloVe...')
    word2vect = pickle.load(open(c.GLOVE_FILEPATH+'.pydict.pkl', 'rb'))

    wnl = WordNetLemmatizer()

    # initialize DB
    cs = CandidateStore(10)
    cs.add_doc('God.txt')
    cs.make_clusters()

    # initialize cache
    cache = OrderedDict()
    cache_capacity = 10

    # start api
    app.run(host='0.0.0.0', port=8000, debug=True)
