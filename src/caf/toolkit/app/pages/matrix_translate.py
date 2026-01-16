from dash import html, register_page

register_page(__name__, path="/matrix-translate", name="Matrix Translate", title="Matrix Translate")

layout = html.Div(
    [
        html.H1("CAF Toolkit - Matrix Zone Translation"),
        html.P("Welcome to caf.toolkit's zone translation functionality for matrices"),
    ]
)
