"""
Useful functions for the image labeller app
"""
import os
import random

from forms import LabelForm
from schema import db_session_scope, User, Image, Label, remove_db_session

IMG_DIR = "static/images"

def fill_user_table(user_name):
    """
    Use the flask session id as the user_id for now
    """
    with db_session_scope() as dbsession:
        user = User(user_name=user_name)
        dbsession.add(user)
        dbsession.commit()
    return True


def fill_image_table_if_empty():
    """
    See if we already have images in the image table - if so
    just return.  If not, loop through the IMG_DIR directory
    and add all images.
    """
    with db_session_scope() as dbsession:
        # see if there are already images in the table
        images = dbsession.query(Image).all()
        if len(images)>0:
            return True
        # image table was empty - fill it now.
        images = os.listdir(IMG_DIR)
        for filename in images:
            image = Image(image_filename=filename)
            dbsession.add(image)
        dbsession.commit()
    return True


def get_user(user_name):
    """
    query the user table for a user_name matching the
    current session_id.
    """
    with db_session_scope() as dbsession:
        user_rows = dbsession.query(User).filter_by(user_name=user_name).all()
        if len(user_rows)==0:
            raise RuntimeError("No user found in db")
        return user_rows[-1].user_id


def get_image(user_id):
    """
    Query the image table for an image, then check that the user has not already labelled
    this image.  (If so, pick another one)
    """
    with db_session_scope() as dbsession:
        image_is_new = False
        image = None
        while not image_is_new:
            images = dbsession.query(Image).all()
            image_index = random.randint(0,len(images)-1)
            image = images[image_index]
            # check if this user has already seen this image
            label_rows = dbsession.query(Label).filter_by(user_id=user_id).\
                                       filter_by(image_id=image.image_id).all()
            image_is_new = len(label_rows)==0
        if image:
            return image.image_filename, image.image_id
        else:
            return None, None



def save_label(user_id, image_id, label, notes):
    """
    Write this label to the database.
    """

    with db_session_scope() as dbsession:
        user = dbsession.query(User).filter_by(user_id=user_id).first()
        image = dbsession.query(Image).filter_by(image_id=image_id).first()
        l = Label(label=label, notes=notes,
                  user=user, image=image)
        dbsession.add(l)
        dbsession.commit()
    return True
