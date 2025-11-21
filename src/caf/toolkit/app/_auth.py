"""Define the authetification endpoints."""

##### IMPORTS #####

import functools
import logging

import wtforms
from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug import security
from wtforms import validators

from ._db import get_db

##### CONSTANTS #####

LOG = logging.getLogger(__name__)

_USERNAME_VALIDATORS = (validators.Length(min=4, max=30),)
_PASSWORD_VALIDATORS = (validators.Length(min=8, max=20),)
_EMAIL_VALIDATORS = (validators.Length(min=6, max=50), validators.Email())

##### CLASSES & FUNCTIONS #####

bp = Blueprint("auth", __name__, url_prefix="/auth")


class RegistrationForm(wtforms.Form):
    username = wtforms.StringField("Username", _USERNAME_VALIDATORS)
    email = wtforms.StringField("Email Address", _EMAIL_VALIDATORS)
    password = wtforms.PasswordField(
        "New Password",
        [
            *_PASSWORD_VALIDATORS,
            validators.EqualTo("confirm_password", message="Passwords must match"),
        ],
    )
    confirm_password = wtforms.PasswordField("Repeat Password")
    admin = wtforms.BooleanField("Admin User")


@bp.route("/register", methods=("GET", "POST"))
def register():
    form = RegistrationForm(request.form)
    if request.method == "POST":
        error = None
        if not form.validate():
            error = "Invalid details"

        db = get_db()

        if error is None:
            try:
                db.execute(
                    "INSERT INTO user (username, password, email, admin) VALUES (?, ?, ?, ?)",
                    (
                        form.username.data,
                        security.generate_password_hash(form.password.data),
                        form.email.data,
                        form.admin.data
                    ),
                )
                db.commit()
            except db.IntegrityError as exc:
                error = str(exc)
            else:
                return redirect(url_for("auth.login"))

        flash(error, category="error")

    return render_template("auth/register.html", form=form)


class LoginForm(wtforms.Form):
    username = wtforms.StringField("Username", _USERNAME_VALIDATORS)
    password = wtforms.PasswordField("Password", _PASSWORD_VALIDATORS)


@bp.route("/login", methods=("GET", "POST"))
def login():
    form = LoginForm(request.form)
    if request.method == "POST":
        error = None
        if not form.validate():
            error = "Invalid, see below"

        if error is None:
            db = get_db()
            user = db.execute(
                "SELECT * FROM user WHERE username = ?", (form.username.data,)
            ).fetchone()

            if user is None:
                error = "Incorrect username."
            elif not security.check_password_hash(user["password"], form.password.data):
                error = "Incorrect password."

        if error is None:
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("index"))

        flash(error, category="error")

    return render_template("auth/login.html", form=form)


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
    else:
        g.user = (
            get_db().execute("SELECT * FROM user WHERE id = ?", (user_id,)).fetchone()
        )


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))

        return view(**kwargs)

    return wrapped_view


def admin_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None or not g.user["admin"]:
            return redirect(url_for("auth.login"))

        return view(**kwargs)

    return wrapped_view
