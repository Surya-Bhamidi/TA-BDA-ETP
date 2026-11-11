"""A Scala source edit must invalidate a previously reusable Spark stage."""
from crime_nlp import audit


def test_scala_source_change_invalidates_stage_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "RAW", tmp_path / "reports.jsonl")
    (tmp_path / "crime_nlp").mkdir()
    (tmp_path / "scala").mkdir()
    for filename in ["settings.toml", "run_pipeline.py", "reports.jsonl"]:
        (tmp_path / filename).write_text("unchanged")
    scala = tmp_path / "scala" / "CrimePipeline.scala"
    scala.write_text("version one")
    before = audit.stage_signature("spark", {"spark_engine": "scala"})
    scala.write_text("version two")
    assert audit.stage_signature("spark", {"spark_engine": "scala"}) != before
