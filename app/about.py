from flask import Blueprint, request, render_template

bp = Blueprint('about', __name__)


@bp.route('/about')
def app():

    return render_template('about.html')
