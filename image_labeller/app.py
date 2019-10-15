"""
Simple flask app for labelling images.

"""
import os
import sys
import random

from flask import Blueprint, Flask, Response, session, \
    render_template, request, redirect, url_for
from flask_cors import CORS
from flask_session import Session
from uuid import uuid4

from forms import LabelForm
from schema import session_scope, User, Image, Label, remove_db_session


IMG_DIR = "static/images"

## Use a flask blueprint rather than creating the app directly
## so that we can also make a test app

blueprint = Blueprint("img-labeller",__name__)


def get_session_id():
    """
    Get the ID from the flask_session Session instance if
    it exists, otherwise just get a default string, which
    will enable us to test some functionality just via python requests.
    """
    if "key" in session.keys():
        return session["key"]
    else:
        return "DEFAULT_SESSION_ID"


def fill_user_table():
    """
    Use the flask session id as the user_id for now
    """
    with session_scope() as session:
        user = User(user_name=get_session_id())
        session.add(user)
        session.commit()
    return True


def fill_image_table():
    images = os.listdir(IMG_DIR)
    with session_scope() as session:
        for filename in images:
            image = Image(image_filename=filename)
            session.add(image)
        session.commit()
    return True


def get_user():
    """
    query the user table for a user_name matching the
    current session_id.
    """
    with session_scope() as session:
        user_rows = session.query(User).filter_by(user_name=get_session_id()).all()
        if len(user_rows)==0:
            raise RuntimeError("No user found in db")
        return user_rows[-1]


@blueprint.teardown_request
def remove_session(ex=None):
    remove_db_session()


@blueprint.route("/")
def homepage():
    """
    basic homepage - options to view results table or run a new test.
    """
    fill_user_table()
    fill_image_table()
    return render_template("index.html")


@blueprint.route("/new",methods=["POST","GET"])
def new_image():
    """
    Display an image, and ask the user to label it
    """
    with session_scope() as session:
        images = session.query(Image).all()
        image_index = random.randint(0,len(images)-1)
        image = images[image_index]
        image_filename = image.image_filename
        image_id = image.image_id
        image_path = os.path.join("static/images",image_filename)
        label_form = LabelForm(request.form)

        if request.method=="POST":
            user = get_user()
            label = label_form.veg_label.data
            notes = label_form.notes.data
            l = Label(label=label, notes=notes,
                      user=user, image=image)
            session.add(l)
            session.commit()
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
