"""Webapp front-end for caf.toolkit built with flask."""

##### IMPORTS #####

import logging
import pathlib

from flask import Flask

from . import _auth, _db, _main, _translate

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    instance_path = pathlib.Path(app.instance_path)

    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=instance_path / "toolkit-app.sqlite",
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    instance_path.mkdir(exist_ok=True, parents=True)
    LOG.debug("Created: %s", instance_path)

    _db.init_app(app)

    app.register_blueprint(_main.bp)
    app.register_blueprint(_auth.bp)
    app.register_blueprint(_translate.bp)
    app.add_url_rule("/", endpoint="index")

    return app
