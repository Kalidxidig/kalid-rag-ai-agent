from app.loaders.pdf_loader import load_pdf
from app.processors.chunker import chunk_text


pdf_path = "documents/elite.pdf"

# Load PDF
text = load_pdf(pdf_path)

print("PDF loaded successfully!")
print("-------------------------")

# Create chunks
chunks = chunk_text(text)

print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print("-------------------------")
print(chunks[0])