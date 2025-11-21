"""Define main app endpoints."""

##### IMPORTS #####

import logging
import pathlib
import re

import wtforms
from flask import Blueprint, flash, g, render_template, request
from werkzeug import utils
from wtforms import validators

from ._auth import login_required

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_UPLOAD_FOLDER = pathlib.Path(".temp/uploads")
_UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)


##### CLASSES & FUNCTIONS #####

bp = Blueprint("translate", __name__)


class TranslateForm(wtforms.Form):
    csv = wtforms.FileField(
        "CSV File",
        [validators.Regexp(r"^[^/\\]+\.(csv|txt)$", re.I, "File should be a CSV")],
    )


@login_required
@bp.route("/translate", methods=("GET", "POST"))
def translate():
    form = TranslateForm(request.form)
    if request.method == "POST":
        if form.validate():
            if form.csv.data not in request.files:
                flash("No file part", category="error")
            else:
                csv_data = request.files[form.csv.data].read()
                (_UPLOAD_FOLDER / utils.secure_filename(form.csv.data)).write_text(
                    csv_data, encoding="utf-8"
                )
                flash(f"Uploaded {form.csv.data}")
        else:
            flash("Invalid", category="error")

    return render_template("translate/translate.html", form=form)
