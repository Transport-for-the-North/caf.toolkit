import base64
import dataclasses
import io
import re
import traceback
from typing import Self

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html, register_page

register_page(__name__, path="/translate", name="Translate", title="Translate")


def upload_widget(
    id_: str, title: str, accepts: str, description: str | None = None
) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title),
            dbc.CardBody(
                children=html.Div(
                    [
                        dcc.Upload(
                            id=f"upload-{id_}",
                            children=html.Div(["Click or drop to upload"]),
                            accept=accepts,
                        ),
                        html.Div(id=f"upload-output-{id_}"),
                    ]
                )
            ),
        ]
    )


def input_widget(
    id_: str, title: str, default: int | str | None = None, description: str | None = None
) -> dbc.Card:
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(dbc.Input(id=f"input-{id_}", value=default))]
    )


file_uploads = [
    upload_widget(
        "data",
        "Upload Data CSV",
        ".csv",
        description="CSV file containing data to be translated",
    ),
    upload_widget(
        "translation",
        "Upload Translation CSV",
        ".csv",
        description="CSV file defining how to translate and the weightings to use",
    ),
]
column_selection = [
    input_widget(
        "from",
        "From Zone Column",
        default=0,
        description="The column (name or position) in the translation",
    ),
    input_widget(
        "to",
        "To Zone Column",
        default=1,
        description="The column (name or position) in the translation"
        " file containing the zone ids to translate to",
    ),
    input_widget(
        "factor",
        "Factor Column",
        default=2,
        description="The column (name or position) in the translation"
        " file containing the weightings between from and to zones",
    ),
]
console = dbc.Card([dbc.CardHeader("Console"), dbc.CardBody(dbc.Textarea("text-console"))])

layout = html.Div(
    [
        html.H1("CAF Toolkit - Zone Translation"),
        html.P("Welcome to caf.toolkit's zone translation functionality test"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([dbc.Col(i) for i in file_uploads]),
                        dbc.Row([dbc.Col(i) for i in column_selection]),
                    ]
                ),
                dbc.Col(console),
            ]
        ),
    ]
)


def extract_content(contents: str):
    matched = re.match(r"data:(\w+)\/(\w+);(\w+),([\w\+\/]+)", contents, re.IGNORECASE)
    if matched is None:
        raise ValueError(f"unknown contents format: {contents[:30]}...")
    return matched.group(1), matched.group(2)


@dataclasses.dataclass
class Contents:
    type_: str
    subtype: str
    contents: bytes
    parameters: str | None = None

    @classmethod
    def parse(cls, text: str) -> Self:
        matched = re.match(r"data:(\w+)\/(\w+)(;.+)*,([\w\+\/=]+)", text, re.IGNORECASE)
        if matched is None:
            raise ValueError(f"unknown contents format: {text[:30]}...")

        decoded = base64.b64decode(matched.group(4))
        return cls(
            type_=matched.group(1),
            subtype=matched.group(2),
            contents=decoded,
            parameters=matched.group(3),
        )


def parse_uploaded_contents(contents: str, filename: str) -> pd.DataFrame:
    if not filename.endswith((".csv", ".txt")):
        raise ValueError(f"invalid file type: {filename.rsplit('.', 1)[1]}")

    content = Contents.parse(contents)
    if content.type_ != "text":
        raise ValueError(f"invalid content type: {content.type_}")
    if content.subtype != "csv":
        raise ValueError(f"invalid content subtype: {content.subtype}")

    file = io.StringIO(content.contents.decode())
    try:
        return pd.read_csv(file)
    except Exception as exc:
        exc.add_note(content.contents.decode()[100:])
        raise


def update_upload_output(id_: str) -> None:
    """Set callback for updating upload output."""

    @callback(
        Output(f"upload-output-{id_}", "children"),
        Input(f"upload-{id_}", "contents"),
        State(f"upload-{id_}", "filename"),
    )
    def update_output(contents: str | None, filename: str | None) -> list:
        if contents is None or filename is None:
            return []
        try:
            data = parse_uploaded_contents(contents, filename)
        except Exception as exc:
            return [
                html.H2(str(filename)),
                html.P(str(exc)),
            ]

        return [
            html.H2(str(filename)),
            dash_table.DataTable(
                data.to_dict("records"),
                columns=[{"name": i, "id": i} for i in data.columns],
                fixed_rows={"headers": True},
                page_size=10,
            ),
        ]


for widget_id in ("data", "translation"):
    update_upload_output(widget_id)
