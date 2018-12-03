from flask import Flask, flash, request, render_template

from candidate_retrieval import CandidateStore
from predictor import Predictor
from multiprocessing import cpu_count

from collections import OrderedDict

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
            candidate_paragraphs = cs.retrieve(question)
            candidate_answers = []
            for context in candidate_paragraphs.values:
                prediction = model.predict(context, question, None, 1)
                candidate_answers.append(prediction[0])

            candidate_answers.sort(key=lambda x: x[1], reverse=True)
            answer = candidate_answers[0][0]
            if len(cache) >= cache_capacity:
                cache.popitem(last=False)
            cache[question] = answer

        return render_template('home.html', answer=answer)


@app.route('/documents')
def documents():
    return render_template('documents.html', files=cs.file_info)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print(request)
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

    # initialize DB
    cs = CandidateStore(10)
    cs.add_doc('app/static/on_method.txt')
    cs.make_clusters()

    # initialize cache
    cache = OrderedDict()
    cache_capacity = 10

    # start api
    app.run(host='0.0.0.0', port=8000, debug=True)
