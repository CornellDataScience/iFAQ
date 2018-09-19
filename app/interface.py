import flask

bp = flask.Blueprint('interface', __name__)

@bp.route('/', methods=['POST'])
def root():
    if request.form['fileIn']:
        print(request.form['fileIn'])
    if request.form['question']:
        print(request.form['question'])
    return flask.render_template('templates/interface/content.html')
