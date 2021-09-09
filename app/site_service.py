from output_create import get_results_and_file
import pandas as pd
from os import path, remove

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, send_file, jsonify
)

bp = Blueprint('/', __name__, url_prefix='/')

@bp.route('/Home', methods=('GET', 'POST'))
def home_page():
    error = None
    yes_no_dict = {"Yes": "3", "No": "2", "NA": "2.17"}
    if request.method == 'POST':
        try:
            remove("static/bar_plot.svg")
        except:
            pass
        manual_features = (yes_no_dict[request.form["one"]], yes_no_dict[request.form["two"]],
                           yes_no_dict[request.form["three"]], request.form["four"],
                           request.form["five"], request.form["six"], request.form["seven"],
                           request.form["eight"], request.form["nine"])
        file = request.files["feature_file"]
        if not file:
            if "" in manual_features:
                error = "Please enter a file or all of the manual feature boxes."
            else:
                get_results_and_file(pd.DataFrame(manual_features).T)
        else:
            get_results_and_file(pd.read_csv(file, header=None))
            file.save("static/DATA.csv")
        plots = None
        if path.isfile("static/bar_plot.svg"):
            plots = "true"
        if error:
            flash(error)
            return render_template('home.html', active='Home', results=None, plots=None)
        return render_template('home.html', active='Home', results="results.csv", plots=plots)
    return render_template('home.html', active='Home', results=None, plots=None)

@bp.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    return jsonify({'ip': request.remote_addr}), 200

@bp.route('/Help')
def help_page():
    return render_template('help.html', active='Help')


@bp.route('/Example')
def example_page():
    return render_template('example.html', active='Example')


@bp.route('/About')
def about_page():
    return render_template('about.html', active='About')

@bp.route('/download-outputs')
def download():
    return send_file("static/results.csv", mimetype="csv", as_attachment=True)

