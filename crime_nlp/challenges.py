"""Explicitly annotated development cases, not a final accuracy benchmark."""
import re

CASES = [
    ("user_international_witness", "Assault", "[[SUSPECT|José García]] hit [[VICTIM|Wei Zhang]] with [[WEAPON|iron rod]] near [[LOCATION|Nairobi]], [[WITNESS|Suresh]] had seen it"),
    ("user_informal_witness", "Robbery", "sir ystrday [[SUSPECT|two fellows]] showed [[WEAPON|knife]] and took my [[PROPERTY|moblie]] near [[LOCATION|bus stand]], [[WITNESS|rahul]] has seen it"),
    ("witness_full_stop", "Assault", "[[SUSPECT|Arjun]] hit [[VICTIM|Meera]] with a [[WEAPON|stick]] near [[LOCATION|Delhi]]. [[WITNESS|Rahul]] saw the incident."),
    ("witness_lowercase", "Assault", "[[SUSPECT|arjun]] punched [[VICTIM|meera]] near [[LOCATION|Pune]], [[WITNESS|suresh]] witnessed the attack."),
    ("witness_uppercase", "Robbery", "[[SUSPECT|TWO MEN]] threatened me with a [[WEAPON|KNIFE]] and took my [[PROPERTY|PHONE]] near [[LOCATION|BUS STAND]], [[WITNESS|RAHUL]] HAS SEEN IT"),
    ("witness_explicit", "Assault", "[[SUSPECT|Kiran]] attacked [[VICTIM|Leena]] near [[LOCATION|Surat]]. Witness [[WITNESS|Nikhil]] told the police what happened."),
    ("passive_roles", "Assault", "[[VICTIM|Meera]] was hit by [[SUSPECT|Arjun]] with a [[WEAPON|stick]] near [[LOCATION|Mumbai]]."),
    ("passive_unknown_names", "Assault", "[[VICTIM|Tenzin Dorje]] was beaten by [[SUSPECT|Chiamaka Okafor]] near [[LOCATION|Nairobi]]."),
    ("active_names", "Assault", "[[SUSPECT|Fatima Al-Zahra]] hit [[VICTIM|Jean-Paul O’Connor]] with a [[WEAPON|stick]] near [[LOCATION|Montréal]]."),
    ("ownership_of", "Theft", "[[SUSPECT|Mayank]] stole the [[PROPERTY|camera]] of [[VICTIM|Surya]] on [[DATE|29-06-2026]]"),
    ("ownership_possessive", "Theft", "[[SUSPECT|Aarav]] stole [[VICTIM|Diya]]'s [[PROPERTY|laptop]] near [[LOCATION|Pune]]."),
    ("ownership_passive", "Theft", "[[VICTIM|Diya]]'s [[PROPERTY|laptop]] was stolen by [[SUSPECT|Aarav]]."),
    ("reported_theft", "Theft", "[[VICTIM|Asha]] reported that her [[PROPERTY|wallet]] was stolen near [[LOCATION|Delhi]]."),
    ("robbery_force", "Robbery", "[[SUSPECT|Ravi]] threatened [[VICTIM|Asha]] with a [[WEAPON|knife]] and took her [[PROPERTY|phone]] near [[LOCATION|Jaipur]]."),
    ("burglary", "Burglary", "An [[SUSPECT|unidentified man]] broke into [[VICTIM|Leena]]'s locked house in [[LOCATION|Jaipur]] and stole a [[PROPERTY|laptop]]."),
    ("arson", "Arson", "[[SUSPECT|Mohan]] deliberately set fire to [[VICTIM|Saira]]'s [[PROPERTY|car]] in [[LOCATION|Bhopal]]."),
    ("vandalism", "Vandalism", "[[SUSPECT|Karan]] smashed [[VICTIM|Aditi]]'s parked [[PROPERTY|bicycle]] near [[LOCATION|Surat]] without taking anything."),
    ("fraud", "Fraud", "A [[SUSPECT|seller]] tricked [[VICTIM|Omar]] into paying an advance for a [[PROPERTY|phone]] that was never delivered."),
    ("cybercrime", "Cybercrime", "An unknown hacker changed Priya's email password and locked her out of the account.", False),
    ("informal_theft", "Theft", "sir my moblie stoln from pocket in bus ystrday", False),
    ("informal_cyber", "Cybercrime", "my acount hackd pasword changed unable login", False),
    ("informal_arson", "Arson", "someone purposely set our scooter on fire", False),
    ("informal_burglary", "Burglary", "went out came home door lock brokn laptop missing inside", False),
    ("evidence_camera", None, "[[EVIDENCE|Camera footage]] was collected."),
    ("evidence_cctv", None, "[[EVIDENCE|CCTV footage]] was reviewed by police."),
    ("evidence_photographs", None, "[[EVIDENCE|Photographs]] were collected by police."),
    ("evidence_screenshots", None, "[[EVIDENCE|Screenshots]] were submitted to investigators."),
    ("evidence_statement", None, "The [[EVIDENCE|witness statement]] was attached to the case file."),
    ("evidence_audio", None, "[[EVIDENCE|Audio recording]] was preserved."),
    ("evidence_negative", None, "No [[EVIDENCE|camera footage]] was available."),
    ("date_time", "Assault", "[[SUSPECT|Arjun]] hit [[VICTIM|Meera]] near [[LOCATION|Pune]] on [[DATE|27.09.2026]] at [[TIME|22:15]]."),
    ("date_slash", "Theft", "[[SUSPECT|Kiran]] stole the [[PROPERTY|wallet]] of [[VICTIM|Neha]] on [[DATE|02/03/2026]]."),
    ("property_camera", "Theft", "[[SUSPECT|Mayank]] stole the [[PROPERTY|camera]] of [[VICTIM|Surya]]."),
    ("property_video_camera", "Theft", "[[SUSPECT|Kiran]] stole [[VICTIM|Meera]]'s [[PROPERTY|video camera]]."),
    ("evidence_after_assault", "Assault", "[[SUSPECT|Rajiv]] hit [[VICTIM|Sneha]] with a [[WEAPON|stick]] near [[LOCATION|Pune]] on [[DATE|29-06-2026]]. A witness called the police. [[EVIDENCE|Camera footage]] was collected."),
    ("witness_name_substitution", "Assault", "[[SUSPECT|Lena Ortiz]] hit [[VICTIM|Noah Reed]] with a [[WEAPON|metal bar]] near [[LOCATION|London]], [[WITNESS|Amara]] had seen it."),
    ("witness_before_event", "Assault", "Witness [[WITNESS|Suresh]] said that [[SUSPECT|José García]] hit [[VICTIM|Wei Zhang]] with an [[WEAPON|iron rod]] near [[LOCATION|Nairobi]]."),
    ("witness_subject", "Assault", "[[WITNESS|Rahul]] saw [[SUSPECT|Ravi]] hit [[VICTIM|Asha]] with a [[WEAPON|stick]] near [[LOCATION|Pune]]."),
]


def render(marked):
    chunks, entities, end, position = [], [], 0, 0
    for match in re.finditer(r"\[\[([A-Z]+)\|(.*?)\]\]", marked):
        literal = marked[end:match.start()]
        chunks.extend([literal, match[2]])
        position += len(literal)
        entities.append({"label": match[1], "text": match[2], "start": position, "end": position + len(match[2])})
        position += len(match[2])
        end = match.end()
    chunks.append(marked[end:])
    return "".join(chunks), entities


def cases():
    for entry in CASES:
        name, category, marked = entry[:3]
        text, entities = render(marked)
        yield {"id": name, "category": category, "text": text, "entities": entities,
               "score_entities": len(entry) < 4 or entry[3]}
