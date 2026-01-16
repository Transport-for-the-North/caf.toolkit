import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, page_container

app = Dash(
    __package__,
    use_pages=True,
    prevent_initial_callbacks=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)


def get_non_home_pages():
    return sorted(
        filter(lambda x: x["path"] != "/", dash.page_registry.values()),
        key=lambda p: p.get("path", ""),
    )


nav_items = []
for page in get_non_home_pages():
    # Give each link a unique id so we can toggle its 'active' property
    link_id = f"navlink-{page['path'] or '/'}".replace("/", "_")
    nav_items.append(
        dbc.NavItem(dbc.NavLink(page["name"], href=page["path"], id=link_id, active=False))
    )


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("CAF.toolkit", href="/"),
            dbc.Nav(
                nav_items, navbar=True, pills=True
            ),  # pills=True gives nice active highlight
            dcc.Location(id="url"),  # watch current URL
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    class_name="mb-4",
)


app.layout = html.Div(
    [
        navbar,
        page_container,  # where Dash Pages renders the active page
    ]
)


# Callback to set the active state based on the URL
# Build Outputs for each nav link dynamically
outputs = [
    dash.Output(f"navlink-{p['path'] or '/'}".replace("/", "_"), "active")
    for p in get_non_home_pages()
]


@app.callback(outputs, dash.Input("url", "pathname"))
def set_active_links(pathname):
    """Mark only the matching nav link as active.

    Use startswith to include top-level and nested pages.
    """
    states = []
    for p in get_non_home_pages():
        path = p["path"] or "/"
        states.append(pathname.startswith(path))
    return states


if __name__ == "__main__":
    app.run(debug=True)
