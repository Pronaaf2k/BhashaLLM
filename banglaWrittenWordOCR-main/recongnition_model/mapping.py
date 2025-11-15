# Update your mapping.py with the correct sizes for Bengali.AI dataset
# mapping.py
grapheme_root_components = list(range(168))  # Should have 168 elements
vowel_diacritic_components = list(range(11))  # Should have 11 elements  
consonant_diacritic_components = list(range(7))  # Should have 7 elements

# Or if you want actual characters, make sure you have enough:
grapheme_root_chars = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 
                      'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ',
                      'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন',
                      'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 
                      'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ'] + [''] * (168 - 46)  # Pad to 168

vowel_diacritic_chars = ['', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ौ']
consonant_diacritic_chars = ['', '্', 'ং', 'ঃ', 'ঁ', '়', '৾']

# Alternative: If you know the exact counts needed, you can use numeric ranges
# grapheme_root_components = list(range(168))
# vowel_diacritic_components = list(range(11)) 
# consonant_diacritic_components = list(range(7))

print(f"✓ mapping.py loaded:")
print(f"  - Grapheme roots: {len(grapheme_root_components)} classes")
print(f"  - Vowel diacritics: {len(vowel_diacritic_components)} classes") 
print(f"  - Consonant diacritics: {len(consonant_diacritic_components)} classes")