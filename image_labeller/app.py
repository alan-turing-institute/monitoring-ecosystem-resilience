"""
Simple flask app for labelling images.

"""
import os
import sys
import random

from flask import Blueprint, Flask, Response, \
    render_template, request, redirect, url_for, session as flask_session
from flask_cors import CORS
from flask_session import Session
from uuid import uuid4

from image_labeller import *
from forms import LabelForm


## Use a flask blueprint rather than creating the app directly
## so that we can also make a test app

blueprint = Blueprint("img-labeller",__name__)


def get_session_id():
    """
    Get the ID from the flask_session Session instance if
    it exists, otherwise just get a default string, which
    will enable us to test some functionality just via python requests.
    """
    if "key" in flask_session.keys():
        return flask_session["key"]
    else:
        return "DEFAULT_SESSION_ID"


@blueprint.teardown_request
def remove_session(ex=None):
    remove_db_session()


@blueprint.route("/")
def homepage():
    """
    basic homepage - options to view results table or run a new test.
    """
    user_name = get_session_id()
    fill_user_table(user_name)
    fill_image_table_if_empty()
    return render_template("index.html")


@blueprint.route("/new",methods=["POST","GET"])
def new_image():
    """
    Display an image, and ask the user to label it
    """
    user_name = get_session_id()
    user_id = get_user(user_name)
    image_filename, image_id = get_image(user_id)
    image_path = os.path.join("static/images",image_filename)
    label_form = LabelForm(request.form)

    if request.method=="POST":
        label = label_form.veg_label.data
        notes = label_form.notes.data
        save_label(user_id, image_id, label, notes)

    # now reset the form to re-render the page
    new_label_form = LabelForm(formdata=None)
    return render_template("new_image.html",
                           veg_image=image_path,
                           img_id=image_id,
                           form=new_label_form)


###########################################

def create_app(name = __name__):
    app = Flask(name)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.secret_key = 'blah'
    CORS(app, supports_credentials=True)
    app.register_blueprint(blueprint)
    Session(app)
    return app


if __name__ == "__main__":

    app = create_app()
    app.run(host='0.0.0.0',port=5002, debug=True)
