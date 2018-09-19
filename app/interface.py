from flask import (
    Blueprint, request, render_template
)

bp = Blueprint('interface', __name__)

@bp.route('/app', methods=['GET', 'POST'])
def app():
    fileIn = request.form['fileIn'] if 'fileIn' in request.files else None
    question = request.form['question'] if 'question' in request.files else None
    if fileIn:
        print(fileIn)
    if question:
        print(question)
    return render_template('interface/content.html')
