import difflib

# --- Configuration ---
# Path to the dictionary file containing valid Bengali words
DICTIONARY_PATH = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\bangla_dictionary.txt"

# --- Load Dictionary ---
def load_dictionary(dictionary_path):
    """
    Loads the dictionary of valid Bengali words from a file.
    :param dictionary_path: Path to the dictionary file.
    :return: List of valid Bengali words.
    """
    print(f"Loading dictionary from: {dictionary_path}")
    try:
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        print(f"Loaded {len(words)} words.")
        return words
    except FileNotFoundError:
        print(f"Error: Dictionary file not found at {dictionary_path}")
        return []

# --- Spelling Correction Function ---
def correct_spelling(input_word, dictionary):
    """
    Corrects the spelling of a given word by finding the closest match in the dictionary.
    :param input_word: The word to correct.
    :param dictionary: List of valid Bengali words.
    :return: Corrected word or the original word if no match is found.
    """
    print(f"Correcting spelling for: {input_word}")
    matches = difflib.get_close_matches(input_word, dictionary, n=1, cutoff=0.8)
    if matches:
        corrected_word = matches[0]
        print(f"Suggested correction: {corrected_word}")
        return corrected_word
    else:
        print("No close match found. Returning original word.")
        return input_word

# --- Main Script ---
if __name__ == "__main__":
    # Load the dictionary
    dictionary = load_dictionary(DICTIONARY_PATH)

    # Example usage
    if dictionary:
        input_word = "অআই"  # Replace with the word you want to correct
        corrected_word = correct_spelling(input_word, dictionary)
        print(f"Corrected Word: {corrected_word}")
    else:
        print("Dictionary not loaded. Cannot perform spelling correction.")