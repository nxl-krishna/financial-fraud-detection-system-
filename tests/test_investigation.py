from streamlit.testing.v1 import AppTest


def test_investigation_renders_without_errors():

    app = AppTest.from_string(
        """
from app.components.investigation import render_investigation

render_investigation()
"""
    )

    app.run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.selectbox) >= 3