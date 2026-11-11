"""Compile Scala with Spark's bundled compiler and submit the actual Scala application.

Python only orchestrates the JVM here; all ETL, SQL and MLlib operations run in Scala.
The installed pyspark package supplies the Spark distribution and Scala compiler.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from .config import ROOT, ARTIFACTS, SETTINGS, CATEGORIES, write_json


def build_scala():
    spark_home = Path(os.environ["SPARK_HOME"])
    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")) if java_home else shutil.which("java")
    if not java:
        raise RuntimeError("Scala Spark requires Java 17. Set JAVA_HOME before running the pipeline.")
    sources = sorted((ROOT / "scala").glob("*.scala"))
    if not sources:
        raise RuntimeError("No Scala application sources found in scala/.")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    jar = ARTIFACTS / "crime-analytics-scala.jar"
    with tempfile.TemporaryDirectory(prefix="scala-classes-", dir=ARTIFACTS) as temporary:
        subprocess.run([java, "-cp", str(spark_home / "jars" / "*"), "scala.tools.nsc.Main", "-usejavacp",
                        "-d", temporary, *map(str, sources)], cwd=ROOT, check=True)
        staged = Path(temporary) / "application.jar"
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(Path(temporary).rglob("*.class")):
                archive.write(file, file.relative_to(temporary).as_posix())
        staged.replace(jar)
    return jar


def submit_scala(class_name, arguments, master="local[2]", log_path=None):
    jar = build_scala()
    spark_home = Path(os.environ["SPARK_HOME"])
    submit = spark_home / "bin" / ("spark-submit.cmd" if os.name == "nt" else "spark-submit")
    command = [str(submit), "--master", master, "--driver-memory", SETTINGS["spark"]["driver_memory"],
               "--class", class_name, str(jar), *map(str, arguments)]
    if log_path is None:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Scala {class_name} failed (exit {result.returncode}); see {log_path}")
    return jar


def run_scala_pipeline(master, raw_path, processed, artifacts):
    processed, artifacts = Path(processed).resolve(), Path(artifacts).resolve()
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    config = {"raw_path": str(Path(raw_path).resolve()), "processed": str(processed), "artifacts": str(artifacts),
              "partitions": SETTINGS["spark"]["partitions"], "categories": CATEGORIES}
    # A private config per invocation avoids concurrent launches sharing arguments.
    with tempfile.TemporaryDirectory(prefix="scala-run-", dir=artifacts) as temporary:
        configuration = Path(temporary) / "config.json"
        write_json(configuration, config)
        jar = submit_scala("CrimePipeline", [configuration], master, artifacts / "scala_pipeline.log")
    report = json.loads((artifacts / "spark_metrics.json").read_text(encoding="utf-8"))
    write_json(artifacts / "scala_pipeline_verification.json", {
        "status": "completed", "language": report["language"], "scala_version": report["scala_version"],
        "spark_version": report["spark_version"], "master": report["master"], "clean_rows": report["clean_rows"],
        "jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
        "source_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                          for file in sorted((ROOT / "scala").glob("*.scala"))}})
    print(f"Scala Spark cleaned {report['clean_rows']:,} reports; held-out MLlib accuracy={report['mllib']['accuracy']:.4f}", flush=True)
    return report
