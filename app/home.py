from flask import Blueprint, request, render_template

bp = Blueprint('home', __name__)


@bp.route('/home', methods=['GET', 'POST'])
def app():
    form = request.form
    if 'question' in form:
        print(form['question'])

    return render_template('home.html')
