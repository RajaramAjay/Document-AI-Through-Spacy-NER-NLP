import re

# Generic cleaning function
def clean_text(value, patterns):
    """Cleans the text by removing specified patterns."""
    cleaned_value = re.sub(patterns, "", value, flags=re.IGNORECASE)
    # cleaned_value = re.sub(r"[^\w\s/]", "", cleaned_value)  # Remove special characters
    return cleaned_value.strip()

# Patterns for cleaning specific fields
CLEANING_PATTERNS = {
    "APPLICANT_NAME": r"(:|FIRST|Last|MIDDLE|NAME|MIDDIEE|MIDDIE|DRIVER|MIDDIT\s*)",

    "DLN": r"(\bDLN\b|DL|:|DRIVER|NO|LICENSE|IDENTIFICATION|NUMBER|DENTIFIC|ATION|LICRNSE|NUMRER|NUMABR|DLN|LICENSF)\s*",

    "DOB": r"(DATE|OF|BIRTH|RIRTH|Gt|NATE|:|DOB|RIRTII|SEX)\s*",

    "DOC_DATE": r"(TODAY'S|:|PRINT|DATE|OF|ISSUE|TORAYS|TODAYS|TONAYS|DATT)\s*",

    "DOC_NAME": r"[\(\)]",

    "CITATION_DATE": r"(CITATION|DATE|:|CONVICTION|NY|REF|ID|COURT|REPORT)\s*",

    "CONVICTION_DATE": r"(CITATION|DATE|:|CONVICTION|NY|REF|ID|COURT|REPORT|CONV)\s*",

    "NY REF ID": r"(CITATION|DATE|:|CONVICTION|NY|REF|ID|COURT|REPORT)\s*",
    
    "COURT REPORT ID": r"(:|ID|ENTIFIER|COURT|REPORT|ACD|CODE|DETAIL)\s*",

    "CONVICTION REASON": r"(REASON|OF|FOR|CONVICTION)\s*",

}

# Function to clean the OCR JSON output
def clean_ocr_json(ocr_json):
    for key, pattern in CLEANING_PATTERNS.items():
        if key in ocr_json:
            ocr_json[key] = clean_text(ocr_json[key], pattern)
    return ocr_json
