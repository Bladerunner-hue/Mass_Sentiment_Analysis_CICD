"""Web forms for main blueprint."""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, TextAreaField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length


class SingleAnalysisForm(FlaskForm):
    """Form for single text sentiment analysis."""

    text = TextAreaField(
        "Text to Analyze",
        validators=[
            DataRequired(message="Please enter some text to analyze"),
            Length(min=1, max=5000, message="Text must be between 1 and 5000 characters"),
        ],
        render_kw={
            "placeholder": "Enter text to analyze sentiment and emotions...",
            "rows": 5,
            "class": "w-full px-4 py-3 border border-gray-300 rounded-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent "
            "resize-none",
        },
    )
    include_emotions = BooleanField(
        "Include Emotion Detection",
        default=True,
        render_kw={"class": "h-4 w-4 text-blue-600 rounded"},
    )
    submit = SubmitField(
        "Analyze",
        render_kw={
            "class": "w-full bg-blue-600 text-white py-3 px-6 rounded-lg "
            "hover:bg-blue-700 transition-colors font-medium"
        },
    )


class BatchUploadForm(FlaskForm):
    """Form for batch CSV upload."""

    file = FileField(
        "CSV File",
        validators=[
            FileRequired(message="Please select a CSV file"),
            FileAllowed(["csv"], message="Only CSV files are allowed"),
        ],
        render_kw={"accept": ".csv", "class": "hidden"},
    )
    include_emotions = BooleanField(
        "Include Emotion Detection",
        default=True,
        render_kw={"class": "h-4 w-4 text-blue-600 rounded"},
    )
    submit = SubmitField(
        "Upload & Process",
        render_kw={
            "class": "w-full bg-green-600 text-white py-3 px-6 rounded-lg "
            "hover:bg-green-700 transition-colors font-medium"
        },
    )


class QuickAnalysisForm(FlaskForm):
    """Form for quick inline analysis."""

    text = StringField(
        "Quick Analysis",
        validators=[DataRequired(), Length(min=1, max=500)],
        render_kw={
            "placeholder": "Type a quick message to analyze...",
            "class": "flex-1 px-4 py-2 border border-gray-300 rounded-l-lg "
            "focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        },
    )
    submit = SubmitField(
        "Analyze",
        render_kw={
            "class": "bg-blue-600 text-white py-2 px-4 rounded-r-lg "
            "hover:bg-blue-700 transition-colors"
        },
    )
