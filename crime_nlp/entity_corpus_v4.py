"""Explicitly rendered ten-label synthetic corpus, with separate scenario families.

This is generator-derived supervision. It is not human annotation or an
independent real-source evaluation. User examples are excluded from templates.
"""
import json
import random
from string import Formatter
import pandas as pd
from .config import SEED
from .annotation_schema import validate_record
from .robust_corpus import NAMES, PLACES, perturb

FIELDS = {"actor": "SUSPECT", "owner": "VICTIM", "observer": "WITNESS", "company": "ORGANIZATION",
          "place": "LOCATION", "destination": "LOCATION", "item": "PROPERTY", "building": "PROPERTY",
          "fuel": "WEAPON", "weapon": "WEAPON", "evidence": "EVIDENCE", "date": "DATE", "time": "TIME"}
# Each split has separately authored event constructions. Shared ontology,
# vocabulary and author still prevent a claim of external independence.
EVENTS = {
    "train": [
        ("Assault", "{actor} hit {owner} with a {weapon} near {place}."),
        ("Assault", "{owner} was attacked by {actor} with a {weapon} outside {place}."),
        ("Theft", "{actor} stole {owner}'s {item} at {place}."),
        ("Theft", "The {item} belonging to {owner} was stolen by {actor}."),
        ("Robbery", "{actor} threatened {owner} with a {weapon} and took their {item}."),
        ("Burglary", "{actor} broke into {owner}'s locked {building} and stole the {item}."),
        ("Arson", "{actor} poured {fuel} and set {owner}'s {building} on fire near {place}."),
        ("Arson", "{actor} used {fuel} from {company} in {place} and used it to burn {owner}'s {building}."),
        ("Fraud", "{actor} tricked {owner} into paying for a {item} using the name of {company}."),
        ("Cybercrime", "{actor} hacked {owner}'s account at {company} and changed its password."),
        ("Vandalism", "{actor} deliberately smashed {owner}'s {item} outside {place}."),
        ("Arson", "{owner}'s {building} was burned by {actor} using {fuel}."),
    ],
    "validation": [
        ("Assault", "During an argument near {place}, {actor} struck {owner} with a {weapon}."),
        ("Theft", "While {owner} was away, {actor} secretly removed their {item}."),
        ("Robbery", "{owner} surrendered the {item} after {actor} showed a {weapon}."),
        ("Burglary", "After forcing a lock, {actor} entered {owner}'s {building} and took a {item}."),
        ("Arson", "{actor} obtained {fuel} from {company} and burned {owner}'s {building} with it."),
        ("Fraud", "{owner} sent money to {actor} for a promised {item}; {company} was falsely named as the seller."),
        ("Cybercrime", "{owner} could no longer access their {company} account after {actor} reset its password."),
        ("Vandalism", "The {item} of {owner} was intentionally damaged by {actor} near {place}."),
    ],
    "development": [
        ("Assault", "{actor} injured {owner} using a {weapon} in {place}."),
        ("Theft", "{owner} discovered that {actor} had taken their {item} without consent."),
        ("Robbery", "Holding a {weapon}, {actor} forced {owner} to hand over the {item}."),
        ("Burglary", "A broken lock let {actor} get inside {owner}'s {building} and steal the {item}."),
        ("Arson", "Using {fuel} bought from {company}, {actor} torched {owner}'s {building}."),
        ("Fraud", "A false offer from {actor}, claiming to work for {company}, cost {owner} money for a {item}."),
        ("Cybercrime", "{actor} accessed {owner}'s {company} login without permission."),
        ("Vandalism", "{owner} reported {actor} for destroying their {item} at {place}."),
    ],
    "test": [
        ("Assault", "In {place}, {owner} suffered an injury when {actor} swung a {weapon} at them."),
        ("Theft", "{actor} walked away with the unattended {item} owned by {owner}."),
        ("Robbery", "A threat involving a {weapon} made {owner} give the {item} to {actor}."),
        ("Burglary", "{owner}'s {building} was unlawfully entered by {actor}, who escaped carrying the {item}."),
        ("Arson", "The {fuel} supplied by {company} was deliberately ignited by {actor}, destroying {owner}'s {building}."),
        ("Fraud", "Under {company}'s name, {actor} deceived {owner} into funding a nonexistent {item}."),
        ("Cybercrime", "An unauthorized password change by {actor} locked {owner} out of {company}."),
        ("Vandalism", "Deliberate blows by {actor} left {owner}'s {item} broken near {place}."),
    ],
}
WITNESSES = {
    "train": [" {observer} saw it.", " {observer} witnessed the incident.", " Witness {observer} told police what happened.", " {observer} has seen it.",
              " Officers took a statement from {observer}, an eyewitness.", " The witness was identified as {observer}.",
              " {observer}, a bystander, saw what happened.", " {observer} gave an eyewitness account.",
              " Police spoke with {observer}, who had observed the event.", " Named witness: {observer}.",
              " An observer, {observer}, was present and saw it.", " {observer} was an eyewitness.",
              " An eyewitness named {observer} described what occurred.", " The person observing the event was {observer}.",
              " The event was witnessed by {observer}.", " Investigators heard from eyewitness {observer}.",
              " The eyewitness who saw it was {observer}.", " This was observed by {observer}, according to the statement.",
              " A witness, {observer}, described the incident."],
    "validation": [" A bystander named {observer} observed the incident.", " Police interviewed witness {observer}."],
    "development": [" The eyewitness was {observer}.", " According to {observer}, who witnessed it, the event occurred nearby."],
    "test": [" An account was provided by eyewitness {observer}.", " The incident was seen by {observer}."],
}


def render(template, values):
    chunks, entities, position = [], [], 0
    for literal, field, _, _ in Formatter().parse(template):
        chunks.append(literal)
        position += len(literal)
        if field:
            value = values[field]
            chunks.append(value)
            if field in FIELDS:
                entities.append({"text": value, "start": position, "end": position + len(value), "label": FIELDS[field]})
            position += len(value)
    return "".join(chunks), entities


def generate(split, count):
    records = []
    split_id = list(EVENTS).index(split)
    for index in range(count):
        rng = random.Random(SEED + 9900001 * (split_id + 1) + index)
        actor, owner, observer = rng.sample(NAMES[split], 3)
        # Most organization names are invented. Train and held-out names do not
        # overlap, and no brand whitelist participates in extraction.
        syllables = [["Meridian", "Copper", "Harbor", "Summit"], ["Orchid", "Cobalt"], ["Cedar", "Aurora"], ["Falcon", "Willow"]][split_id]
        suffixes = ["Energy", "Bank", "Retail", "Networks", "Services", "Trading", "Oil", "Petroleum", "Fuels", "Telecom"]
        invented = "".join(rng.choice(["al", "mar", "ve", "no", "ki", "ra", "su", "te"]) for _ in range(3)).title()
        company = f"{rng.choice(syllables)} {invented} {rng.choice(suffixes)}" if index % 3 == 0 else f"{rng.choice(syllables)} {rng.choice(suffixes)}"
        values = {"actor": actor, "owner": owner, "observer": observer, "company": company,
                  "place": rng.choice(PLACES[split]), "destination": rng.choice(PLACES[split]),
                  "item": rng.choice(["phone", "laptop", "wallet", "camera", "bicycle", "office bag", "identity cards"]),
                  "building": rng.choice(["flat", "shop", "garage", "warehouse", "house", "store"]),
                  "fuel": rng.choice(["petrol", "kerosene", "gasoline", "diesel", "paint thinner", "lighter fluid"]),
                  "weapon": rng.choice(["knife", "stick", "iron rod", "pistol", "metal bar"]),
                  "evidence": rng.choice(["camera footage", "CCTV recording", "photographs", "witness statement", "screenshots"]),
                  "date": f"{rng.randrange(1,29):02}/{rng.randrange(1,13):02}/2026", "time": f"{rng.randrange(24):02}:{rng.randrange(60):02}"}
        category, event = EVENTS[split][index % len(EVENTS[split])]
        template = event
        if index % 5 != 0:
            template += rng.choice(WITNESSES[split])
        if "{company}" not in template and index % 3 == 0:
            template += " The nearby business was {company}."
        if index % 3 == 1:
            template += " Police collected {evidence}."
        if index % 4 == 0:
            template += " The report was recorded on {date} at {time}."
        text, entities = render(template, values)
        text, entities = perturb(text, entities, rng, rng.choice([0, 0, 0, 1, 4, 5]))
        row = {"report_id": f"V4-{split}-{index:06}", "narrative": text, "crime_type": category,
               "template_group": f"v4-{split}-{index % len(EVENTS[split])}", "split": split,
               "source": "synthetic-explicit-schema-v4", "entities_json": json.dumps(entities, ensure_ascii=False),
               "annotation_schema": "4.0", "annotation_provenance": "Explicit generator fields; no human annotation"}
        validate_record(row)
        records.append(row)
    return pd.DataFrame(records)
