from streamlit.testing.v1 import AppTest


def test_overview_renders_without_errors():

    app = AppTest.from_string(
        """
from app.components.overview import render_overview

render_overview()
"""
    )

    app.run(timeout=20)

    assert len(app.exception) == 0