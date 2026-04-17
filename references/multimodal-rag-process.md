Google Embedding Models

1. gemini-embedding-001 (only for texts)
2. gemini-embedding-2-preview (has multimodal embedding capability - works with images also)

# What should we do when docling finds different content types of PDF ?

```
PDF File
   │
   ▼
┌─────────────────────────────┐
│  Docling Parser             │  → Extracts text, tables, images
│  (docling_parser.py)        │
└──────────────┬──────────────┘
               │
               ▼ detects content type
       ┌───────┴────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
  ┌─────────┐             ┌──────────┐             ┌──────────┐
  │  Text   │             │  Table   │             │  Image   │
  └────┬────┘             └────┬─────┘             └────┬─────┘
       │                       │                        │
       ▼                       ▼               ┌────────┴────────┐
  Extract Text            Extract Table        │                 │
       │                       │           Option A          Option B
       ▼                       ▼               │                 │
   Chunk Text             Convert to        Convert to       Upload to
                          DataFrame         Base64 img       S3 / Cloud
                              │                  │                 │
                              ▼                  └────────┬────────┘
                          Chunk Table                     ▼
                                            Vision-capable LLM
                                            (e.g. Gemini Pro Vision)
                                            → Generate image description
       │                       │                         │
       ▼                       ▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Generate Embeddings                       │
│                                                              │
│   Text / Table / Image desc →  gemini-embedding-2-preview    │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Vector Database     │
                    │      (PgVector)       │
                    └───────────────────────┘
```

**Key decisions:**

- **Text** → chunk → embed with `gemini-embedding-001`
- **Table** → DataFrame → chunk → embed with `gemini-embedding-001`
- **Image** → store as Base64 or S3 URL → Vision LLM generates a description → embed with `gemini-embedding-2-preview`

> Images are never stored raw in the vector DB. Their LLM-generated descriptions are, enabling semantic search over visual content.
