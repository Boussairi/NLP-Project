import re

class EnglishTextCleaning():

    def __init__(self) :
        pass

    def remove_links(self, text):
        # Expression régulière pour trouver et enlever les liens
        link_pattern = re.compile(r'http\S+|www\S+')
        return link_pattern.sub(r'', text)


    def extract_hashtags(self, df, text_column_name, new_column_name):
        # Create an empty list to store extracted hashtags
        hashtags_list = []
        
        # Regular expression pattern to match hashtags
        hashtag_pattern = r'#\w+'
        
        # Loop through each row in the DataFrame
        for text in df[text_column_name]:
            # Find all hashtags in the text using regular expression
            hashtags = re.findall(hashtag_pattern, text)
            # Append the list of hashtags to the hashtags_list
            hashtags_list.append(hashtags)
        
        # Add the hashtags_list as a new column in the DataFrame
        df[new_column_name] = hashtags_list
        
        return df


    def remove_hashtags(self, text):
        cleaned_text = re.sub(r'#\w+', '', text)
        return cleaned_text


    # Fonction pour enlever les caractères spéciaux
    def remove_special_chars(self, text):
        # Expression régulière pour trouver les caractères spéciaux 
        special_chars_pattern = re.compile(r'[^\w\s]')
        return re.sub(special_chars_pattern, '', text)


    def remove_spaces_and_newlines(self, text):
        # Remplace les espaces, les tabulations et les sauts de ligne par une chaîne vide
        return text.replace("\t", "").replace("\n", "")


    def extract_emojis(self, text):
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        return emoji_pattern.findall(text)