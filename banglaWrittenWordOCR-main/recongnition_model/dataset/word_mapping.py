# word_mapping.py
# Comprehensive Bangla character mapping for OCR

# Define the character mapping for Bangla words
mapping = [
    # Basic Bengali vowels and consonants
    'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 
    'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 
    'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 
    'ত', 'থ', 'দ', 'ধ', 'ন', 
    'প', 'ফ', 'ব', 'ভ', 'ম', 
    'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ',
    'ড়', 'ঢ়', 'য়', 'ৎ',
    
    # Bengali vowel diacritics (matras)
    'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ',
    
    # Bengali modifiers
    '্',  # Virama (hasant)
    'ং',  # Anusvara
    'ঃ',  # Visarga
    'ঁ',  # Chandrabindu
    
    # Digits
    '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯',
    
    # Special characters and punctuation
    ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`', 
    '{', '|', '}', '~',
    
    # Additional common characters
    '্', '৳', '₹'  # Currency symbols
]

# You can also add these utility functions if needed:

def get_char_index(char):
    """Get index of a character in the mapping"""
    try:
        return mapping.index(char)
    except ValueError:
        return -1  # Return -1 for unknown characters

def get_char_from_index(index):
    """Get character from index in the mapping"""
    if 0 <= index < len(mapping):
        return mapping[index]
    return '?'  # Return '?' for invalid indices

def is_valid_char(char):
    """Check if a character exists in the mapping"""
    return char in mapping

# Print info when module is loaded
print(f"✓ word_mapping loaded successfully!")
print(f"✓ Total characters in mapping: {len(mapping)}")
print(f"✓ First 10 characters: {mapping[:10]}")
print(f"✓ Last 10 characters: {mapping[-10:]}")