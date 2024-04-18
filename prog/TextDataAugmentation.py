import random

class TextDataAugmentation:
    def __init__(self, synonyms_dict):
        self.synonyms_dict = synonyms_dict
    
    def synonym_replacement(self, text, n=1):
        words = text.split()
        new_words = words.copy()
        random.shuffle(new_words)
        for _ in range(n):
            for i, word in enumerate(words):
                if word.lower() in self.synonyms_dict:
                    synonym = random.choice(self.synonyms_dict[word.lower()])
                    new_words[i] = synonym
        return ' '.join(new_words)
    
    def random_insertion(self, text, n=1):
        words = text.split()
        for _ in range(n):
            word = random.choice(list(self.synonyms_dict.keys()))
            synonym = random.choice(self.synonyms_dict[word])
            idx = random.randint(0, len(words))
            words.insert(idx, synonym)
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        words = text.split()
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        return ' '.join(new_words)

# Example usage
synonyms_dict = {
    'happy': ['joyful', 'content', 'pleased'],
    'sad': ['unhappy', 'gloomy', 'miserable'],
    'angry': ['irate', 'furious', 'outraged']
}
text_augmentor = TextDataAugmentation(synonyms_dict)

text = "The quick brown fox jumps over the lazy dog."
augmented_text = text_augmentor.synonym_replacement(text)
print("Synonym Replacement:", augmented_text)

augmented_text = text_augmentor.random_insertion(text)
print("Random Insertion:", augmented_text)

augmented_text = text_augmentor.random_deletion(text)
print("Random Deletion:", augmented_text)
