from streamlit.testing.v1 import AppTest


def test_model_monitoring_renders_without_errors():

    app = AppTest.from_string(
        """
from app.components.model_monitoring import render_model_monitoring

render_model_monitoring()
"""
    )

    app.run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.metric) >= 10