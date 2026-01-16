from dash import html, register_page

register_page(__name__, path="/", name="Home", title="Home")

layout = html.Div(
    [html.H1("CAF Toolkit"), html.P("Welcome to caf.toolkit's web-app which contains...")]
)
