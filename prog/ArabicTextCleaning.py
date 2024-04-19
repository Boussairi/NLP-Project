import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from alphabet_detector import AlphabetDetector
import string
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AdamW
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
#import preProcessData #type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
import pickle



class ArabicTextCleaning:
    def __init__(self, path): 
        self.path = path
    def remove_diacritics(self,text):
        arabic_diacritics = re.compile("""
                                ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)
        return text


    def remove_punctuation(self, s):
        my_punctuations = string.punctuation + "،" + "؛" + "؟" + "«" + "»"
        translator = str.maketrans('', '', my_punctuations)
        return s.translate(translator)


    def replace_punctuation(self ,s): # replace punctuation with space
        my_punctuations = string.punctuation + "،" + "؛" + "؟" + "«" + "»"
        replace_table = str.maketrans(my_punctuations,  ' '*len(my_punctuations))
        return s.translate(replace_table)



    def remove_links(self, text):
        # return re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', ' ', text, flags=re.MULTILINE).strip()
        # return re.sub(r'^https?:\/\/.*[\r\n]*', '', clean_text, flags=re.MULTILINE)
        return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)


    def remove_empty_lines(self, text):
        lines = [s.rstrip() for s in text.split("\n") if s.rstrip()]
        return '\n'.join(lines)



    def keep_only_arabic(self, text):
        ad = AlphabetDetector()
        clean_lines = list()
        for line in text.splitlines():
            clean_line = list()
            for word in line.split():
                if len(word) > 1:
                    if ad.is_arabic(word):
                        if word.isalpha():
                            clean_line.append(word)
            clean_lines.append(' '.join(clean_line))
        return '\n'.join(clean_lines)


    def clean_doc(self, text):
        text = text.replace('(', ' ')
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text) # remove numbers
        text = text.replace(')', ' ')
        clean_text = self.remove_links(text)
        clean_text = self.remove_diacritics(clean_text)
        clean_text = self.replace_punctuation(clean_text)
        clean_text = self.keep_only_arabic(clean_text)
        clean_text = self.remove_empty_lines(clean_text)
        return clean_text

    def translate_dialect(self,data): 
        mapping_dict = {'nile': 'النيل',
      'magreb': 'المغرب',
      'msa': 'الفصحى',
      'gulf': 'الخليج',
      'levant': 'الشام'}

        data['dialect'] = data['dialect'].replace(mapping_dict)


    def load_data(self): 
        data = pd.read_csv(self.path)
        return data
    
    def drop_columns(self,data):  
        data = data.drop(['rephrase', 'id'], axis=1)
        return data
    
    def split_data(self, X, y): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

