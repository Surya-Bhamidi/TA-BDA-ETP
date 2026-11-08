"""Additional annotated variations, kept distinct from the original repair audit."""
from .challenges import render

CASES = [
    ("comma_witness", "Assault", "[[SUSPECT|Amir Khan]] hit [[VICTIM|Lina Chen]] with a [[WEAPON|rod]] near [[LOCATION|Nairobi]], [[WITNESS|Dev]] saw it."),
    ("semicolon_witness", "Assault", "[[SUSPECT|Ravi]] hit [[VICTIM|Asha]] near [[LOCATION|Pune]]; [[WITNESS|Neha]] had seen it."),
    ("witness_newline", "Assault", "[[SUSPECT|Ravi]] hit [[VICTIM|Asha]] near [[LOCATION|Pune]].\n[[WITNESS|Neha]] witnessed it."),
    ("witness_multiword", "Assault", "[[SUSPECT|Ravi]] hit [[VICTIM|Asha]] near [[LOCATION|Pune]], [[WITNESS|Neha Sharma]] has seen it."),
    ("witness_first", "Assault", "[[WITNESS|Neha]] saw [[SUSPECT|Ravi]] hit [[VICTIM|Asha]] near [[LOCATION|Pune]]."),
    ("witness_title", "Assault", "[[SUSPECT|Ravi]] hit [[VICTIM|Asha]] near [[LOCATION|Pune]]. Witness [[WITNESS|Neha]] reported the incident."),
    ("knife_mobile", "Robbery", "[[SUSPECT|Two men]] showed a [[WEAPON|knife]] and took my [[PROPERTY|mobile]] near the [[LOCATION|bus stand]], [[WITNESS|Rahul]] saw it."),
    ("camera_stolen", "Theft", "[[SUSPECT|Amir]] stole [[VICTIM|Lina]]'s [[PROPERTY|camera]] near [[LOCATION|Delhi]]."),
    ("camera_evidence", None, "[[EVIDENCE|Camera recordings]] were obtained by police."),
    ("dashcam_evidence", None, "[[EVIDENCE|Dashcam footage]] was examined by investigators."),
    ("screenshot_evidence", None, "[[EVIDENCE|Screenshots]] were preserved."),
    ("passive_unseen_names", "Assault", "[[VICTIM|Lina Chen]] was hit by [[SUSPECT|Amir Khan]] with a [[WEAPON|rod]] near [[LOCATION|Nairobi]]."),
    ("lowercase_passive", "Assault", "[[VICTIM|asha]] was hit by [[SUSPECT|ravi]] with a [[WEAPON|stick]] near [[LOCATION|pune]]."),
    ("ownership_new_names", "Theft", "[[SUSPECT|Amir Khan]] stole the [[PROPERTY|wallet]] of [[VICTIM|Lina Chen]]."),
    ("passive_ownership", "Theft", "[[VICTIM|Lina]]'s [[PROPERTY|wallet]] was stolen by [[SUSPECT|Amir]]."),
    ("damage_property", "Vandalism", "[[SUSPECT|Ravi]] smashed [[VICTIM|Asha]]'s [[PROPERTY|bicycle]] near [[LOCATION|Pune]]."),
    ("titled_observer", "Assault", "Witness [[WITNESS|Neha]] saw [[SUSPECT|Ravi]] hit [[VICTIM|Asha]] near [[LOCATION|Pune]]."),
]


def cases():
    for name, category, marked in CASES:
        text, entities = render(marked)
        yield {"id": name, "category": category, "text": text, "entities": entities, "score_entities": True}
