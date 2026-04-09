"""Temporary script to explore Docling API."""
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    generate_picture_images=True,
)

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert("data/RIL-Media-Release-RIL-Q2-FY2024-25-Financial-and-Operational-Performance.pdf")
doc = result.document

items = list(doc.iterate_items())
print(f"Total items: {len(items)}")

for i, item in enumerate(items[:8]):
    print(f"\n--- Item {i} ---")
    print(f"  type: {type(item).__name__}")
    if isinstance(item, tuple):
        level, node = item
        print(f"  level: {level}")
        print(f"  node type: {type(node).__name__}")
        print(f"  node label: {getattr(node, 'label', 'N/A')}")
        text = getattr(node, "text", "")
        if text:
            print(f"  text: {text[:120]}")
        prov = getattr(node, "prov", None)
        if prov:
            print(f"  prov: {prov}")
    else:
        print(f"  label: {getattr(item, 'label', 'N/A')}")
        text = getattr(item, "text", "")
        if text:
            print(f"  text: {text[:120]}")

# Check for tables and pictures specifically
print("\n\n=== CONTENT TYPE SUMMARY ===")
type_counts = {}
for item in items:
    if isinstance(item, tuple):
        _, node = item
    else:
        node = item
    label = str(getattr(node, "label", type(node).__name__))
    type_counts[label] = type_counts.get(label, 0) + 1
for label, count in sorted(type_counts.items()):
    print(f"  {label}: {count}")

# Check table export
print("\n\n=== TABLE EXPLORATION ===")
for item in items:
    if isinstance(item, tuple):
        _, node = item
    else:
        node = item
    label = str(getattr(node, "label", ""))
    if "table" in label.lower():
        print(f"Table node type: {type(node).__name__}")
        print(f"Table attrs: {[a for a in dir(node) if not a.startswith('_')]}")
        if hasattr(node, "export_to_markdown"):
            md = node.export_to_markdown()
            print(f"Table markdown:\n{md[:300]}")
        break

# Check picture/image export
print("\n\n=== PICTURE EXPLORATION ===")
for item in items:
    if isinstance(item, tuple):
        _, node = item
    else:
        node = item
    label = str(getattr(node, "label", ""))
    if "picture" in label.lower() or "image" in label.lower():
        print(f"Picture node type: {type(node).__name__}")
        print(f"Picture attrs: {[a for a in dir(node) if not a.startswith('_')]}")
        if hasattr(node, "image"):
            img = node.image
            print(f"Image type: {type(img)}")
        if hasattr(node, "get_image"):
            img = node.get_image(doc)
            print(f"get_image type: {type(img)}")
        break
