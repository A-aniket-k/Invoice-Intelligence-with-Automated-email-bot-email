import os
import json
from google import genai
from PIL import Image
import pypdfium2 as pdfium

def extract_invoice_data(uploaded_file) -> dict:
    """
    Handles PDF/Images extraction via Gemini Vision API.
    """
    file_name = uploaded_file.name.lower()
    client = genai.Client()
    
    if file_name.endswith('.pdf'):
        try:
            pdf = pdfium.PdfDocument(uploaded_file.read())
            page = pdf[0]
            bitmap = page.render(scale=2)
            img = bitmap.to_pil()
        except Exception as e:
            print(f"Error converting PDF page to image: {e}")
            return None
    else:
        img = Image.open(uploaded_file)
    
    prompt = """
    You are an expert financial audit agent. Look at this invoice and extract the following numeric values.
    Return ONLY a raw JSON object with these exact keys, using 0 or null if a value is completely missing.

    Required Keys:
    - invoice_quantity
    - invoice_dollars
    - Freight
    - total_item_quantity
    - total_item_dollars
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[img, prompt]
    )
    
    try:
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        extracted_json = json.loads(cleaned_text)
        return extracted_json
    except Exception as e:
        print(f"Error parsing model output to JSON: {e}")
        return None