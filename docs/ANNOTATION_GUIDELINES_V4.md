# Annotation contract 4.0

Annotate exact source character spans. Start is inclusive; end is exclusive.
Do not rewrite spelling, capitalization, accents, whitespace or punctuation.
Exclude leading articles and honorifics and possessive apostrophes from people.
Keep internal initials, apostrophes and hyphens. BIO training requires spans to
be non-overlapping. Annotate each mention separately.

| Label | Meaning and boundary |
|---|---|
| SUSPECT | Person or described group explicitly alleged to perform the incident action. An officer, relative, supplier or reporting person is not automatically a suspect. |
| VICTIM | Person affected by the incident, including an explicitly named owner of stolen/damaged property. |
| WITNESS | Person explicitly described as observing the event or designated its witness/eyewitness. Someone merely submitting a report need not be a witness. |
| ORGANIZATION | Named company, bank, institution or supplier, including its name suffix. Physical locality/address mentions are LOCATION. If a particular mention explicitly identifies an organization as the accused actor, use the event role for that mention and flag the ontology limitation for adjudication. |
| LOCATION | Named or specific physical scene, source locality, address or destination. Keep address components belonging to one mention together. |
| WEAPON | Instrument used or explicitly mentioned as a potential instrument of harm, including an accelerant used for a damaging fire. Benign fuel purchases are not weapon use. |
| PROPERTY | Tangible item or premises explicitly possessed, taken, damaged or targeted; include meaningful compound names such as “office bag,” exclude its owner's separate name. |
| EVIDENCE | Explicit evidence material, recordings, samples or statements collected/reviewed/submitted in the case context. A supplier is not itself evidence. |
| DATE | Complete date expression, retaining original separators or relative phrasing. |
| TIME | Complete time expression, retaining colon, period and meridiem where present. |

One object can have different roles in different mentions. “Petrol was stolen”
is PROPERTY; petrol used to burn a flat is WEAPON/accelerant; petrol residue
collected for testing is EVIDENCE. A pronoun with an ambiguous antecedent must
be flagged for review rather than resolved by guessing.

Record assertion separately from entity type: mentioned, negated, alleged,
disputed or uncertain. A named person still exists in the text when the action
is denied. The assertion concerns the described role/action, not proof of guilt.
Quoted and reported claims should retain attribution. Where one span would
need multiple incompatible roles, adjudicate the mention in its event context;
do not create overlapping BIO annotations.

## Human review and real-data protocol

1. Record permission, source identifier and source group before importing.
2. Two annotators work on the unchanged source independently, using pseudonymous
   reviewer IDs. Model suggestions are drafts and must not be silently treated
   as human gold.
3. Run the agreement tool. Inspect disagreements in boundaries, type and
   assertion; adjudicate them with a reviewer and rationale.
4. Deduplicate and split by source group before fitting. Keep related cases and
   near duplicates together. The tool rejects exact normalized duplicates;
   humans must also check semantic near duplicates.
5. Freeze the adjudicated test version and access it once after selection. If
   its results inform further changes, report subsequent runs as diagnostics
   and obtain another independent test for new accuracy claims.

No permissioned real-data evaluation has been performed. The supplied v4
experiments use synthetic examples, including disclosed rule-assisted migration
of old training references. Legacy benchmark labels and weights are preserved.
