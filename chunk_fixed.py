def fixed_character_splitter(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and < chunk_size"
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks


document = "Your long document goes here..." * 100

chunks = fixed_character_splitter(document)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {len(chunk)} characters")
    print(chunk[:100])