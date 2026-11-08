# Fictional demonstration inputs

`demo_case.txt` is a plain-text input for Narrative lab or the console demo.
`demo_reports.jsonl` deliberately contains one valid fictional record and one
invalid record with blank text, for demonstrating ingestion validation and
quarantine. It contains no real police records.

```powershell
.\.venv312\Scripts\python.exe scripts\demo_project.py --file examples\demo_case.txt
.\.venv312\Scripts\python.exe scripts\import_jsonl.py examples\demo_reports.jsonl --source-url "local-demo://examples/demo_reports.jsonl" --license "Project-generated fictional demonstration text"
```

The importer stages one accepted row and one rejection separately under
`data/external/`. It does not change the 60,000-report benchmark or dashboard.
