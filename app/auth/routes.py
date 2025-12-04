"""Authentication routes."""

from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user

from app.auth import bp
from app.auth.forms import LoginForm, RegistrationForm, ChangePasswordForm
from app.extensions import db
from app.models.user import User


@bp.route("/login", methods=["GET", "POST"])
def login():
    """User login page."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = LoginForm()

    if form.validate_on_submit():
        email = form.email.data.lower().strip()
        password = form.password.data

        user = User.get_by_email(email)

        if user is None or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("auth/login.html", form=form)

        if not user.is_active:
            flash("Your account has been disabled.", "error")
            return render_template("auth/login.html", form=form)

        # Log in the user
        login_user(user, remember=form.remember_me.data)
        user.update_last_login()
        db.session.commit()

        flash("Welcome back!", "success")

        # Redirect to next page or dashboard
        next_page = request.args.get("next")
        if next_page and next_page.startswith("/"):
            return redirect(next_page)
        return redirect(url_for("main.dashboard"))

    return render_template("auth/login.html", form=form)


@bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = RegistrationForm()

    if form.validate_on_submit():
        user = User(username=form.username.data.strip(), email=form.email.data.lower().strip())
        user.set_password(form.password.data)

        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", form=form)


@bp.route("/logout")
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.index"))


@bp.route("/profile")
@login_required
def profile():
    """User profile page."""
    return render_template("auth/profile.html", user=current_user)


@bp.route("/profile/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    """Edit user profile."""
    from wtforms import StringField
    from wtforms.validators import DataRequired, Email, Length
    from flask_wtf import FlaskForm

    class EditProfileForm(FlaskForm):
        username = StringField("Username", validators=[DataRequired(), Length(min=3, max=64)])
        email = StringField("Email", validators=[DataRequired(), Email()])

    form = EditProfileForm(obj=current_user)

    if form.validate_on_submit():
        # Check for duplicate username
        if form.username.data != current_user.username:
            existing = User.get_by_username(form.username.data)
            if existing:
                flash("Username is already taken.", "error")
                return render_template("auth/edit_profile.html", form=form)

        # Check for duplicate email
        if form.email.data.lower() != current_user.email:
            existing = User.get_by_email(form.email.data)
            if existing:
                flash("Email is already registered.", "error")
                return render_template("auth/edit_profile.html", form=form)

        current_user.username = form.username.data.strip()
        current_user.email = form.email.data.lower().strip()
        db.session.commit()

        flash("Profile updated successfully.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("auth/edit_profile.html", form=form)


@bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    """Change password page."""
    form = ChangePasswordForm()

    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            flash("Current password is incorrect.", "error")
            return render_template("auth/change_password.html", form=form)

        current_user.set_password(form.new_password.data)
        db.session.commit()

        flash("Password changed successfully.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("auth/change_password.html", form=form)


@bp.route("/api-key")
@login_required
def api_key():
    """API key management page."""
    return render_template("auth/api_key.html", user=current_user)


@bp.route("/api-key/generate", methods=["POST"])
@login_required
def generate_api_key():
    """Generate new API key."""
    current_user.generate_api_key()
    db.session.commit()

    flash("New API key generated. Store it securely!", "success")
    return redirect(url_for("auth.api_key"))


@bp.route("/api-key/revoke", methods=["POST"])
@login_required
def revoke_api_key():
    """Revoke API key."""
    current_user.revoke_api_key()
    db.session.commit()

    flash("API key revoked.", "success")
    return redirect(url_for("auth.api_key"))
