from flask import Blueprint, request, render_template

bp = Blueprint('main', __name__)


@bp.route('/app', methods=['GET', 'POST'])
def app():
    form = request.form
    files = request.files 
    if 'question' in form:
        print(form['question'])
    if 'fileIn' in form and files is not None:
        print(form['fileIn'])
        print(type(files))

    return render_template('content.html')
