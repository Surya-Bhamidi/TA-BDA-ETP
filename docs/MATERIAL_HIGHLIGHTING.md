# Highlighting petrol and similar materials

In the reported narrative, **petrol is highlighted as WEAPON** with the contextual instrument type `accelerant`:

> Abebe Bekele poured petrol and set Olga Ivanova's shop on fire near Dubai. kindly take my complaint

The project uses WEAPON for instruments used in the alleged offence, including accelerants. The dashboard explains this meaning beneath the narrative. This is an entity role in the text, not a legal determination.

The informal generator originally contained literal “petrol” without an entity placeholder, so its reference annotations omitted the word. The learned model therefore often omitted it too. The fix adds a disclosed material-role check after model extraction. Checkpoint weights, label counts and existing reference annotations are unchanged. Corrections retain exact text and offsets, record their extraction method and have null model confidence.

| Context | Result |
|---|---|
| Petrol poured and used to set a shop on fire | WEAPON, accelerant |
| Kerosene, gasoline, diesel, paint thinner or lighter fluid linked to starting a damaging fire | WEAPON, accelerant |
| Fuel stolen from someone | PROPERTY |
| Fuel samples/residue collected for examination, or explicitly seized as evidence | EVIDENCE |
| Buying fuel, filling a car, safe storage, stove use or a controlled campfire | No new WEAPON/EVIDENCE label from this check |
| An unrelated person starts a fire, or the described use is explicitly denied | No new accelerant claim from this check |

The rule follows grammatical links between the material and its action, with narrowly bounded action-sequence patterns for all-capital text and unfamiliar names that confuse the parser. Those patterns allow dotted initials in names while retaining sentence, negation and different-actor safeguards. A fire mentioned elsewhere does not automatically turn every fuel mention into a weapon. It preserves already correct learned containing spans such as “petrol can” and their model scores.

## Checks and limits

There are 37 explicit material-role development checks. All pass through Automatic/CRF and through BERT with the context layer. Neutral checks prohibit false WEAPON/EVIDENCE labels; they do not certify every other prediction. The initial 21-case check passed 9 cases before the fix. These examples informed development and are not a real-world accuracy benchmark.

### Supplier and reused fuel example

> Suresh used petrol from Indian Oil in Vanasthalipuram and used it to burn Ramesh's flat in Hasthinapuram

Both CRF and BERT with the current context layer return Suresh: SUSPECT;
petrol: WEAPON/accelerant; Vanasthalipuram: LOCATION; Ramesh: VICTIM;
flat: PROPERTY; Hasthinapuram: LOCATION. The schema-4 trained role specialist
now also highlights Indian Oil as ORGANIZATION. It is the named supplier and
does not receive a participant or weapon label.

The material check can follow one coordinated “used it to burn” reference when
the actor is unchanged and no intervening object competes with the fuel.
Negated reuse and a different explicit actor are guarded against. This is
bounded reference handling, not a general coreference model. Clear arguments
of that linked fire action are checked too, including the owned property and
the localities of the supplier and fire. Tests cover the complete seven-entity
set with both taggers and its dashboard display.

The complete refreshed corpus is checked for source offsets and overlapping mentions. The specific `poured petrol and set` case-library reports are checked for WEAPON predictions and working entity search. Dashboard tests check the rendered petrol highlight and explanatory caption.

The legacy eight-label context evaluation is recorded in `artifacts/narrative_context_validation.json`. At that revision, material highlighting changed CRF diagnostic Span-F1 from 92.40% to 92.21%, and BERT from 80.79% to 80.63%; CRF validation was 93.93%. Added petrol mentions can be meaningful yet absent from generator gold. The schema-4 live pipeline has separate ten-label diagnostics in `hybrid_entity_evaluation.json`; see UPGRADE_STATUS.md. Raw legacy model metrics are unchanged. Other materials, misspellings and difficult grammar may still be missed.

## Run locally

```powershell
$projectPython = '.\.venv312\Scripts\python.exe'
& $projectPython scripts\check_material_context.py --output artifacts\material_context_after.json
& $projectPython scripts\check_material_context.py --ner BERT --output artifacts\material_context_bert.json
& $projectPython scripts\demo_project.py --file examples\petrol_arson.txt --quick
& $projectPython -m pytest -q
```

After a material-rule update, close the dashboard and run
`& $projectPython scripts\refresh_material_entities.py` to rerun selected raw
NER and context checks for material narratives. The script preserves other
entity rows and verifies their contents and report order. The broader
`run_pipeline.py --stage enrich --entities-only` command refreshes all reports.

In the dashboard, choose Narrative lab, paste the reported narrative and analyze it. The existing WEAPON filter can find stored petrol instruments in Case explorer.
