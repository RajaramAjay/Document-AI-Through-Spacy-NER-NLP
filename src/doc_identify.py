import re

def identify_document_type(recognized_text):
    dc_pattern = re.compile(r"DC[-\s]?\d{3,4}")  # Pattern to identify DC-type documents
    keywords = ["Health Questionnaire", "Driver Wellness & Safety", "MOTOR VEHICLE ADMINISTRATION"]
    score = 0
    document_name = None
    
    for text in recognized_text:
        # Check for the DC pattern
        if dc_pattern.search(text):
            score += 2  # Higher score for DC pattern match
            document_name = dc_pattern.search(text).group()  # Capture the document name (e.g., "DC-001")

        # Check for keywords and add to score
        score += sum(1 for keyword in keywords if keyword in text)
    
    # Use boolean to indicate if the document is a DC-type
    is_dc_type = score >= 2
    
    return is_dc_type, document_name