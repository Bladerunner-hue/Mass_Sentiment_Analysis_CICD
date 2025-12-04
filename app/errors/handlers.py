"""Error handlers for the application."""

from flask import render_template, request, jsonify


def wants_json_response():
    """Check if the client prefers JSON response."""
    return (
        request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html
    ) or request.path.startswith("/api/")


def handle_400(error):
    """Handle 400 Bad Request errors."""
    if wants_json_response():
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": (
                        str(error.description) if hasattr(error, "description") else "Bad request"
                    ),
                }
            ),
            400,
        )

    return render_template("errors/400.html", error=error), 400


def handle_401(error):
    """Handle 401 Unauthorized errors."""
    if wants_json_response():
        return jsonify({"error": "unauthorized", "message": "Authentication required"}), 401

    return render_template("errors/401.html", error=error), 401


def handle_403(error):
    """Handle 403 Forbidden errors."""
    if wants_json_response():
        return (
            jsonify(
                {
                    "error": "forbidden",
                    "message": "You do not have permission to access this resource",
                }
            ),
            403,
        )

    return render_template("errors/403.html", error=error), 403


def handle_404(error):
    """Handle 404 Not Found errors."""
    if wants_json_response():
        return (
            jsonify({"error": "not_found", "message": "The requested resource was not found"}),
            404,
        )

    return render_template("errors/404.html", error=error), 404


def handle_500(error):
    """Handle 500 Internal Server errors."""
    if wants_json_response():
        return (
            jsonify({"error": "internal_server_error", "message": "An unexpected error occurred"}),
            500,
        )

    return render_template("errors/500.html", error=error), 500


def handle_validation_error(error):
    """Handle Marshmallow validation errors."""
    return (
        jsonify(
            {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": error.messages,
            }
        ),
        400,
    )
