"""
Flask app for labelling images.
"""

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import subprocess
import os
import sys
import time
import random
from forms import LabelForm


app = Flask(__name__)
#app.config.from_object(config.Config)

@app.route("/")
def homepage():
    """
    basic homepage - options to view results table or run a new test.
    """
    return render_template("index.html")


@app.route("/new",methods=["POST","GET"])
def new_image():
    """
    Display an image, and ask the user to label it
    """
    images = os.listdir("static/images")
    image_index = random.randint(0,len(images)-1)
    image = os.path.join("static/images",images[image_index])
    label_form = LabelForm(request.form)

    if request.method=="POST":
        label = label_form.veg_label.data
        notes = label_form.notes.data
        print(label, notes)

    return render_template("new_image.html", veg_image=image, form=label_form)




if __name__ == "__main__":
    app.run(host='0.0.0.0')
