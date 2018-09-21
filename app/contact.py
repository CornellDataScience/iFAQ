from flask import Blueprint, request, render_template

bp = Blueprint('contact', __name__)


@bp.route('/contact')
def app():

    return render_template('contact.html')
