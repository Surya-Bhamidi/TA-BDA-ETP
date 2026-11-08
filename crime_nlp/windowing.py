"""Tokenizer-bounded source windows retaining original character offsets."""


def source_windows(texts, tokenizer, max_tokens=128, overlap=32):
    if not texts:
        return []
    if max_tokens < 8 or not 0 <= overlap < max_tokens - 2:
        raise ValueError("Invalid token window or overlap")
    encoded = tokenizer(list(texts), truncation=True, max_length=max_tokens, stride=overlap,
                        return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
    result = []
    for row, offsets in enumerate(encoded["offset_mapping"]):
        source_id = int(encoded["overflow_to_sample_mapping"][row])
        content = [(first, last) for first, last in offsets if last > first]
        if not content:
            continue
        first, last = content[0][0], content[-1][1]
        result.append({"source": source_id, "start": first, "end": last, "text": texts[source_id][first:last]})
    return result
