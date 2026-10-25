from __future__ import annotations

import json
import threading

import numpy as np

from models import Alert, Target
from storage import FlightStorage


def test_layout_and_records(tmp_path, settings_factory):
    s = settings_factory(data_dir=tmp_path)
    st = FlightStorage(s)
    for sub in ("frames", "crops"):
        assert (st.root / sub).is_dir()
    for k in ("detections", "matches", "alerts", "telemetry"):
        assert (st.root / f"{k}.jsonl").exists()
    img = np.zeros((40, 60, 3), np.uint8)
    rel = st.save_frame(img, "000001")
    assert rel and (tmp_path / rel).is_file() and rel.endswith(".jpg")
    assert st.url_for(rel) == "/data/" + rel
    assert st.url_for(tmp_path / rel) == "/data/" + rel
    crop = st.save_crop(img, "c1")
    assert crop and "crops" in crop
    a = Alert(level="person", track_id=1, ts=1.0, c_yolo=0.9)
    assert st.append("alerts", a)
    assert json.loads((st.root / "alerts.jsonl").read_text().splitlines()[0])["track_id"] == 1
    st.record_target(Target(description="red jacket"))
    st.close()
    meta = json.loads((st.root / "flight.json").read_text())
    assert meta["targets"][0]["description"] == "red jacket" and meta["end_ts"]


def test_thread_safe_appends(tmp_path, settings_factory):
    st = FlightStorage(settings_factory(data_dir=tmp_path))

    def w(i):
        for j in range(50):
            st.append("telemetry", {"i": i, "j": j})
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = (st.root / "telemetry.jsonl").read_text().splitlines()
    assert len(lines) == 200 and all(json.loads(x) for x in lines)


def test_io_errors_do_not_raise(tmp_path, settings_factory):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    st = FlightStorage(settings_factory(data_dir=blocker))   # cannot mkdir under a file
    assert not st.ok
    assert st.save_frame(np.zeros((4, 4, 3), np.uint8), "x") is None
    assert st.append("alerts", {"a": 1}) is False
    st.close()


def test_url_for_with_relative_data_dir(tmp_path, monkeypatch, settings_factory):
    """DATA_DIR=data (relative, as start.sh runs it): crop URLs must be /data/flights/..., not /data/data/..."""
    from storage import FlightStorage
    monkeypatch.chdir(tmp_path)
    fs = FlightStorage(settings_factory(data_dir="data"))
    crop = fs.root / "crops" / "t1.jpg"
    assert not str(crop).startswith("/")
    url = fs.url_for(crop)
    assert url.startswith("/data/flights/") and "/data/data/" not in url
    assert fs.url_for("flights/x/crops/a.jpg") == "/data/flights/x/crops/a.jpg"
