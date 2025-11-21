"""Define main app endpoints."""

##### IMPORTS #####

import logging

from flask import Blueprint, render_template

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####

bp = Blueprint("main", __name__)


@bp.route("/", methods=("GET",))
def index():
    return render_template("index.html")
