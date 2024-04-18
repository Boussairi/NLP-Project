import random
def random_swap(text, num_swaps=1):
    # Tokenize the text into words
    words = text.split()
    
    # Perform word swaps
    swapped_texts = []
    if len(words) >= 2:
        for _ in range(num_swaps):
            # Copy the original list to prevent modifying the original text
            swapped_words = words[:]
            idx_1, idx_2 = random.sample(range(len(words)), 2)
            swapped_words[idx_1], swapped_words[idx_2] = swapped_words[idx_2], swapped_words[idx_1]
            swapped_texts.append(' '.join(swapped_words))
    
    return swapped_texts

def random_deletion(text, p=0.5):
    # Tokenize the text into words
    words = text.split()
    
    # Perform random deletion
    remaining_texts = []
    for _ in range(len(words)):
        remaining_words = [word for idx, word in enumerate(words) if random.uniform(0, 1) > p or idx == _]
        # If all words are deleted, keep at least one word
        if len(remaining_words) == 0:
            remaining_words = [random.choice(words)]
        remaining_texts.append(' '.join(remaining_words))
    
    return remaining_texts

def augment_data_with_RS_RD(data, sarcastic_column='sarcastic', swap_p=0.5, deletion_p=0.5, num_swaps=1):
    # Separate sarcastic and non-sarcastic tweets
    sarcastic_data = data[data[sarcastic_column] == 1]
    non_sarcastic_data = data[data[sarcastic_column] == 0]
    
    # Determine the number of sarcastic tweets needed to balance the data
    num_sarcastic_needed = len(non_sarcastic_data) - len(sarcastic_data)
    
    # Perform random swap and random deletion on sarcastic tweets to balance the data
    augmented_rows = []
    while num_sarcastic_needed > 0:
        for index, row in sarcastic_data.iterrows():
            # Apply random_swap function to the 'text' column
            swapped_texts = random_swap(row['text'], num_swaps=num_swaps)
            
            # Apply random_deletion function to each swapped text
            for swapped_text in swapped_texts:
                deleted_texts = random_deletion(swapped_text, p=deletion_p)
                
                # Create a new row for each deleted text and append to the list
                for deleted_text in deleted_texts:
                    augmented_row = row.copy()
                    augmented_row['text'] = deleted_text
                    augmented_rows.append(augmented_row)
                    num_sarcastic_needed -= 1
                    if num_sarcastic_needed == 0:
                        break
            if num_sarcastic_needed == 0:
                break
    
    # Combine the original non-sarcastic tweets with the augmented sarcastic tweets
    augmented_data = pd.concat([non_sarcastic_data, pd.DataFrame(augmented_rows)], ignore_index=True)
    
    return augmented_data
