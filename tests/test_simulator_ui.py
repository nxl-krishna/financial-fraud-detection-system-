
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.components import simulator


def test_simulator_renders_without_errors(monkeypatch):

    sample_data = pd.DataFrame({
        "Time": [100000.0, 100060.0, 100120.0],
        "Amount": [100.0, 250.0, 500.0],
        "Class": [0, 1, 0],
    })

    for i in range(1, 29):
        sample_data[f"V{i}"] = [0.0, 0.1, -0.1]

    monkeypatch.setattr(
        simulator,
        "load_simulation_data",
        lambda: sample_data.copy(),
    )

    monkeypatch.setattr(
        simulator,
        "check_api_health",
        lambda: {
            "online": True,
            "model": "Random Forest",
        },
    )

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
