"""Authentication forms."""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError, Regexp

from app.models.user import User


class LoginForm(FlaskForm):
    """User login form."""

    email = StringField(
        "Email",
        validators=[
            DataRequired(message="Email is required"),
            Email(message="Please enter a valid email address"),
        ],
        render_kw={
            "placeholder": "you@example.com",
            "type": "email",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "autocomplete": "email",
        },
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(message="Password is required")],
        render_kw={
            "placeholder": "Your password",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "autocomplete": "current-password",
        },
    )
    remember_me = BooleanField("Remember me", render_kw={"class": "h-4 w-4 text-blue-600 rounded"})
    submit = SubmitField(
        "Sign In",
        render_kw={
            "class": "w-full bg-blue-600 text-white py-3 px-6 rounded-lg "
            "hover:bg-blue-700 transition-colors font-medium"
        },
    )


class RegistrationForm(FlaskForm):
    """User registration form."""

    username = StringField(
        "Username",
        validators=[
            DataRequired(message="Username is required"),
            Length(min=3, max=64, message="Username must be 3-64 characters"),
            Regexp(
                r"^[a-zA-Z0-9_]+$",
                message="Username can only contain letters, numbers, and underscores",
            ),
        ],
        render_kw={
            "placeholder": "Choose a username",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "autocomplete": "username",
        },
    )
    email = StringField(
        "Email",
        validators=[
            DataRequired(message="Email is required"),
            Email(message="Please enter a valid email address"),
        ],
        render_kw={
            "placeholder": "you@example.com",
            "type": "email",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "autocomplete": "email",
        },
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(message="Password is required"),
            Length(min=8, message="Password must be at least 8 characters"),
        ],
        render_kw={
            "placeholder": "Create a strong password",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "autocomplete": "new-password",
        },
    )
    password2 = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(message="Please confirm your password"),
            EqualTo("password", message="Passwords must match"),
        ],
        render_kw={
            "placeholder": "Confirm your password",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "autocomplete": "new-password",
        },
    )
    submit = SubmitField(
        "Create Account",
        render_kw={
            "class": "w-full bg-green-600 text-white py-3 px-6 rounded-lg "
            "hover:bg-green-700 transition-colors font-medium"
        },
    )

    def validate_username(self, field):
        """Check if username is already taken."""
        user = User.get_by_username(field.data)
        if user:
            raise ValidationError("Username is already taken.")

    def validate_email(self, field):
        """Check if email is already registered."""
        user = User.get_by_email(field.data)
        if user:
            raise ValidationError("Email is already registered.")


class ChangePasswordForm(FlaskForm):
    """Change password form."""

    current_password = PasswordField(
        "Current Password",
        validators=[DataRequired()],
        render_kw={
            "placeholder": "Your current password",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        },
    )
    new_password = PasswordField(
        "New Password",
        validators=[
            DataRequired(),
            Length(min=8, message="Password must be at least 8 characters"),
        ],
        render_kw={
            "placeholder": "Your new password",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        },
    )
    new_password2 = PasswordField(
        "Confirm New Password",
        validators=[DataRequired(), EqualTo("new_password", message="Passwords must match")],
        render_kw={
            "placeholder": "Confirm your new password",
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        },
    )
    submit = SubmitField(
        "Change Password",
        render_kw={
            "class": "w-full bg-blue-600 text-white py-3 px-6 rounded-lg "
            "hover:bg-blue-700 transition-colors font-medium"
        },
    )
