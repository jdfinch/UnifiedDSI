import nltk
from nltk.stem import WordNetLemmatizer
import pathlib as pl

lemmatizer = None

plural_exceptions = {
    'media'
}

def depluralize(word):
    """
    Converts a plural noun to its singular form using NLTK's WordNetLemmatizer.
    
    :param word: The plural word to depluralize
    :return: The singular form of the word
    """
    global lemmatizer
    if lemmatizer is None:
        # Download WordNet data if not already done
        if not pl.Path('~/nltk_data/corpora/wordnet.zip').expanduser().exists():
            nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
    # Lemmatize the word as a noun
    lower_word = word.lower()
    if lower_word in plural_exceptions: 
        singular = lower_word
    else:
        singular = lemmatizer.lemmatize(lower_word, pos='n')
    if word[:1].isupper():
        singular = singular[:1].upper()+singular[1:]
    return singular

def untokenize_text(input_string):
    for punc in '.!?,':
        input_string = input_string.replace(f" {punc}", f"{punc}")
    for suffix in ('ly', 'er', 's',):
        input_string = input_string.replace(f" -{suffix}", suffix)
    input_string = input_string.replace(' s ', 's ')
    return input_string

def textify(
    input_string, 
    strip_non_alpha=True, 
    underscore_to_space=True, 
    camelcase_to_space=True, 
    plural_to_singular=False,
    add_prefix=None
):
    """
    Standardizes an input string based on specified options.

    Parameters:
        input_string (str): The string to be standardized.
        strip_non_alpha (bool): If True, removes non-alphabetic characters.
        underscore_to_space (bool): If True, replaces underscores with spaces.
        camelcase_to_space (bool): If True, adds spaces before uppercase letters in camelCase.
        plural_to_singular (bool): If True, converts the last word to singular form
        add_prefix (str or None): If not None, prepends this string to the input string.

    Returns:
        str: The standardized string.
    """
    # Add a prefix if provided
    if add_prefix:
        input_string = f"{add_prefix} {input_string}"

    # Replace underscores with spaces
    if underscore_to_space:
        input_string = input_string.replace('_', ' ')

    # Remove non-alphabetic characters
    if strip_non_alpha:
        input_string = input_string.replace('-', ' ')
        input_string = ''.join(c for c in input_string if c.isalpha() or c.isspace())
        input_string = ' '.join(input_string.split())  # Remove extra spaces

    # Convert camelCase to space-separated words
    if camelcase_to_space:
        chars = [input_string[:1]]  # Start with the first character
        for i, c in enumerate(input_string[1:], start=1):
            if c.isupper() and input_string[i - 1].islower():
                chars.extend((' ', c.lower()))
            else:
                chars.append(c)
        input_string = ''.join(chars)

    if plural_to_singular:
        words = input_string.split()
        words = words[:-1] + [depluralize(words[-1])]
        input_string = ' '.join(words)

    return input_string




if __name__ == "__main__":
    print(f"{textify(input_string='cats_dogs', strip_non_alpha=True, underscore_to_space=True, camelcase_to_space=True, plural_to_singular=True, add_prefix=None) = }")
    print(f"{textify(input_string='berriesLeaves', strip_non_alpha=True, underscore_to_space=True, camelcase_to_space=True, plural_to_singular=True, add_prefix='domain') = }")
    print(f"{textify(input_string='example_string', strip_non_alpha=True, underscore_to_space=True, camelcase_to_space=False, plural_to_singular=True, add_prefix=None) = }")
    print(f"{textify(input_string='GeeseAndMice', strip_non_alpha=True, underscore_to_space=True, camelcase_to_space=True, plural_to_singular=True, add_prefix=None) = }")
    print(f"{textify(input_string='slot_name', strip_non_alpha=True, underscore_to_space=True, camelcase_to_space=False, plural_to_singular=False, add_prefix='domain') = }")

