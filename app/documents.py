from flask import Blueprint, request, render_template

bp = Blueprint('documents', __name__)


@bp.route('/documents', methods=['GET', 'POST'])
def app():
    form = request.form
    files = request.files
    if 'fileIn' in form and files is not None:
        print(form['fileIn'])
        print(type(files))

    return render_template('documents.html')
