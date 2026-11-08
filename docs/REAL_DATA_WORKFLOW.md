# Prepare an independently reviewed dataset

This workflow is ready for permissioned narratives supplied by the user. It does
not manufacture real data or claim that reviewer metadata proves independence.

Use ANNOTATION_GUIDELINES_V4.md. Each JSONL record contains:

```json
{"report_id":"source-a-001","narrative":"Mira saw the incident.","source_group":"source-a","permission":"permission reference","reviewer":"reviewer-2","review_status":"adjudicated","entities":[{"text":"Mira","start":0,"end":4,"label":"WITNESS","assertion":"mentioned"}]}
```

Record categories, relation annotations and assertion qualifiers when available.
Source groups should represent independent case/source families rather than
arbitrary names assigned merely to produce three splits. At least three groups
are required. Keep private material in ignored data/external; access remains local.

```powershell
$projectPython = '.\.venv312\Scripts\python.exe'
# Compare two independently prepared versions, before adjudication.
& $projectPython scripts\annotation_workflow.py agreement data\external\annotator_a.jsonl data\external\annotator_b.jsonl --output artifacts\annotation_agreement.json
# After reviewing disagreements, prepare source-separated files in a new folder.
& $projectPython scripts\annotation_workflow.py split data\external\adjudicated.jsonl --output data\external\reviewed_splits_v1
```

The split command rejects duplicate IDs, normalized duplicate narratives,
invalid spans, missing source/permission metadata and records not marked as
adjudicated. It preserves original text. It never merges real records into the
synthetic corpus or modifies existing benchmark references.

## Dashboard feedback is a draft queue

Open Analysis details → Correct extracted entities. Edit offsets and labels,
then save. By default the database retains only offsets, labels and a text hash.
The unchecked “Include this narrative in the local annotation queue” option
explicitly opts into saving source text locally for later human review.

```powershell
& $projectPython scripts\export_entity_feedback.py --output data\external\feedback_drafts_v1.jsonl
```

Exported drafts still require source grouping, permission, independent review
and adjudication. They are not automatically retrained into the model and must
not be mixed into a locked test split.
