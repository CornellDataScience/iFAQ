from flask import Flask, request, render_template

from predictor import Predictor
from multiprocessing import cpu_count

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        question = request.form['question']
        predictions = model.predict('Sean is a legend', question, None, 1)
        answer = predictions[0]

        return render_template('home.html', label=answer)


if __name__ == '__main__':
    # load ml model
    model = Predictor(
        'app/m_reader.mdl',
        normalize=True,
        embedding_file=None,
        char_embedding_file=None,
        num_workers=cpu_count() // 2
    )
    # start api
    app.run(host='0.0.0.0', port=8000, debug=True)
