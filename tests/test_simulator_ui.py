from streamlit.testing.v1 import AppTest


def test_simulator_renders_without_errors():

    app = AppTest.from_string(
        """
from app.components.simulator import render_simulator

render_simulator()
"""
    )

    app.run(timeout=30)

    assert len(app.exception) == 0
    assert len(app.button) >= 1
    assert len(app.selectbox) >= 1