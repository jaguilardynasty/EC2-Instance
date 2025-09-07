import torch
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import asyncio
# from get_embeddings import get_text_embeddings_from_list  # Ensure this is correctly implemented

# Assuming model and tokenizer are loaded outside of functions for efficiency
# model = joblib.load("embedd_wide_net_terms.joblib")
# vectorizer = joblib.load("vectorizer_embedd_wide_net_terms.joblib")
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")       distillargenet
# bert_model = DistilBertForSequenceClassification.from_pretrained("/Users/aguilar/Desktop/distilbert_model_2", local_files_only=True)

model = joblib.load("embedds_wide_net_pP.joblib")
vectorizer = joblib.load("vectorizer_embedds_wide_net_pP.joblib")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# bert_model = DistilBertForSequenceClassification.from_pretrained("/Users/aguilar/Desktop/distilbert_pP_fine_tuned", local_files_only=True)
bert_model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert_pP_fine_tuned_2", local_files_only=True)

model2 = joblib.load("embedd_wide_net_terms.joblib")
vectorizer2 = joblib.load("vectorizer_embedd_wide_net_terms.joblib")
bert_model2 = DistilBertForSequenceClassification.from_pretrained("./models/distilbert_TermsModel_scary_11", local_files_only=True)


bert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def replace_multiple_newlines(strings):
    replaced_strings = []
    for string in strings:
        replaced_string = re.sub(r'\n{2,}', '\n\n', string)
        replaced_strings.append(replaced_string)
    return replaced_strings


def filter_sentences_above_threshold(sentences, model, vectorizer, threshold=0.0):

   
    # Check if the list of sentences is empty
    if not sentences:
        return []

    vectorized_sentences = vectorizer.transform(sentences)
    probabilities = model.predict_proba(vectorized_sentences)[:, 1]  # Assuming index 1 is "dangerous"

    sent_w_probs = [sent for sent, prob in zip(sentences, probabilities) if prob > threshold]

    # if len(sent_w_probs) > 300:
    #     sent_w_probs = [sent for sent, prob in zip(sentences, probabilities) if prob > (threshold + 0.21)]

    # if len(sent_w_probs) > 250:
    #     sent_w_probs = [sent for sent, prob in zip(sentences, probabilities) if prob > (threshold + 0.14)]


    # if len(sent_w_probs) > 200:
    #     sent_w_probs = [sent for sent, prob in zip(sentences, probabilities) if prob > (threshold + 0.07)]

    # if len(sent_w_probs) > 200:
    #     sent_w_probs = [sent for sent, prob in zip(sentences, probabilities) if prob > (threshold + 0.3)]

    return sent_w_probs

def create_chunks(long_sentence, chunk_size=32, step=10):
    words = long_sentence.split()
    chunks = []
    for i in range(0, len(words)):
        end_index = i + chunk_size
        if end_index > len(words):
            break  # Stop if the next chunk extends beyond the sentence length
        chunk_words = words[i:end_index]
        # Adjust for ellipses
        next_step_index = i + step
        if next_step_index < len(words):
            next_chunk_words = words[next_step_index:next_step_index + chunk_size]
            if len(next_chunk_words) < 20:
                # Extend current chunk to include words to meet the minimum chunk length
                chunk_words += next_chunk_words
        chunk = " ".join(chunk_words)
        if len(chunk.split()) >= 20:  # Ensure chunk length is 15 words or more
            chunks.append(chunk)
    return chunks
  
def chunkify_flags(sentences, tokenizer, model, device, batch_size=1):
    flagged_sentences = []
    model = model.to(device)

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        encoded = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = F.softmax(outputs.logits, dim=1)
            predicted_classes = torch.argmax(outputs.logits, dim=1)

            for j, label in enumerate(predicted_classes):
                sentence_prob = probabilities[j][1].item()
                if label == 1 or (label == 0 and sentence_prob >0.3):
                    sentence = batch_sentences[j]
                    # Check if the sentence needs to be chunked
                    if len(sentence.split()) > 65:
                        chunks = create_chunks(sentence)
                        chunk_probs = []
                        for chunk in chunks:
                            # Re-process each chunk
                            chunk_encoded = tokenizer([chunk], padding=True, truncation=True, return_tensors="pt").to(device)
                            with torch.no_grad():
                                chunk_output = model(**chunk_encoded)
                                chunk_probabilities = F.softmax(chunk_output.logits, dim=1)
                                chunk_sentence_prob = chunk_probabilities[0][1].item()
                                chunk_probs.append((chunk, chunk_sentence_prob))
                                # print(chunk)
                        # Find the highest probability chunk if exists
                        if chunk_probs:
                            highest_prob_chunk = max(chunk_probs, key=lambda x: x[1])
                            flagged_sentences.append((highest_prob_chunk[0], sentence, sentence_prob))
                    else:
                        flagged_sentences.append((batch_sentences[j], batch_sentences[j]))

    return flagged_sentences


def process_sentences_with_bert(sentences, tokenizer, model, device, batch_size=1):
    model = model.to(device)
    print(f"Processing {len(sentences)} sentences with BERT")
    flags_to_return = []
    flags = set()
    with open('all_flags.txt', 'r') as file:
        all_sentences = file.read().splitlines()

    for i in range(len(sentences)):
        if(len(sentences[i]) < 10):
            continue
        if sentences[i] in all_sentences:
            flags_to_return.append(("\"" + sentences[i] + "\"", sentences[i], 0.95)) # Call the callback with the flagged sentence
            print("AUTOMATIC ONE CAUGHT")
            print(f"Flagged sentence: {sentences[i]} with probability: {0.95}")
            flags.add(sentences[i])
            continue
           


        
    
        batch_sentence = [sentences[i]]
        encoded = tokenizer(batch_sentence, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = F.softmax(outputs.logits, dim=1)
            predicted_classes = torch.argmax(outputs.logits, dim=1)

            for j, label in enumerate(predicted_classes):
                sentence_prob = probabilities[j][1].item()
                current_sentence_index = i

                if label == 1 or (label ==0 and sentence_prob >0.3):
                    sentence = batch_sentence[j]

                    # Initialize variables for combined sentence probabilities
                    combined_prev_prob = 0
                    combined_next_prob = 0

                    # Check the previous sentence combined with the current one
                    if current_sentence_index == 0: #if current_sentence_index > 0: 
                        prev_sentence = sentences[current_sentence_index - 1]
                        combined_prev = prev_sentence + " " + sentence
                        combined_prev_encoded = tokenizer([combined_prev], padding=True, truncation=True, return_tensors="pt").to(device)
                        combined_prev_prob = F.softmax(model(**combined_prev_encoded).logits, dim=1)[0][1].item()
                        print(f"Combined previous sentence probability: {combined_prev_prob} for: {combined_prev}")

                    # Check the current sentence combined with the next one
                    if current_sentence_index < 0:#len(sentences) - 1:  # Ensure we have a next sentence
                        next_sentence = sentences[current_sentence_index + 1]
                        combined_next = sentence + " " + next_sentence
                        print(f"Combined next sentence (before encoding): {combined_next}")
                        combined_next_encoded = tokenizer([combined_next], padding=True, truncation=True, return_tensors="pt").to(device)
                        combined_next_prob = F.softmax(model(**combined_next_encoded).logits, dim=1)[0][1].item()
                        print(f"Combined next sentence probability: {combined_next_prob} for: {combined_next}")

                    # Determine the highest probability
                    best_prob = sentence_prob
                    best_sentence = sentence

                    if combined_prev_prob > best_prob:
                        best_prob = combined_prev_prob
                        best_sentence = combined_prev

                    if combined_next_prob > best_prob:
                        best_prob = combined_next_prob
                        best_sentence = combined_next

                    print(f"Best sentence: {best_sentence} with probability: {best_prob}")

                    # Check if the sentence needs to be chunked
                    if len(best_sentence.split()) > 40:
                        chunks = create_chunks(best_sentence)
                        chunk_probs = []
                        for chunk in chunks:
                            # Re-process each chunk
                            chunk_encoded = tokenizer([chunk], padding=True, truncation=True, return_tensors="pt").to(device)
                            with torch.no_grad():
                                chunk_output = model(**chunk_encoded)
                                chunk_probabilities = F.softmax(chunk_output.logits, dim=1)
                                chunk_sentence_prob = chunk_probabilities[0][1].item()
                                chunk_probs.append((chunk, chunk_sentence_prob))
                        # Find the highest probability chunk if exists
                        if chunk_probs:
                            highest_prob_chunk = max(chunk_probs, key=lambda x: x[1])
                            if highest_prob_chunk[1] > 0.0:
                                if highest_prob_chunk[0] not in flags:
                                    print(f"Highest probability chunk: {highest_prob_chunk[0]} with probability: {highest_prob_chunk[1]}")
                                    flags.add(highest_prob_chunk[0])

                                    if highest_prob_chunk[0][0] != best_sentence[0] and highest_prob_chunk[0][-1] != best_sentence[-1]:
                                        flags_to_return.append(("\"..." + highest_prob_chunk[0] + "...\"", best_sentence, highest_prob_chunk[1]))
                                    elif highest_prob_chunk[0][0] != best_sentence[0]:
                                        flags_to_return.append(("\"..." + highest_prob_chunk[0] + "\"", best_sentence, highest_prob_chunk[1]))
                                    elif highest_prob_chunk[0][-1] != best_sentence[-1]:
                                        flags_to_return.append(("\"" + highest_prob_chunk[0] + "...\"", best_sentence, highest_prob_chunk[1]))

                    else:
                        if best_sentence not in flags:
                            if best_prob > 0.0:
                                flags_to_return.append(("\"" + best_sentence + "\"", best_sentence, best_prob))  # Call the callback with the flagged sentence
                                print(f"Flagged sentence: {best_sentence} with probability: {best_prob}")
                        flags.add(best_sentence)
    
    return flags_to_return

# Ensure to integrate this function properly in your Flask app as shown before.

# Ensure to integrate this function properly in your Flask app as shown before.


def create_chunks(text, max_length=40):
    words = text.split()
    for i in range(0, len(words), max_length):
        if i + (i + max_length) >= 20:
            yield ' '.join(words[i:i + max_length])



# async def group_sentences_based_on_embeddings(sentences_with_scores, get_embeddings_func, similarity_threshold=0.8, high_probability_bonus=0.035):
#     if not sentences_with_scores:  # Check if the list is empty
#         return []

#     # Obtain embeddings for all sentences
#     sentences, orig_sentences, scores = zip(*sentences_with_scores)
#     embeddings = await get_embeddings_func(sentences)
#     used = set()
#     groups = []

#     for i, embedding_i in enumerate(embeddings):
#         if i in used:
#             continue
#         # Start a new group with the current sentence and its score
#         current_group = [(sentences[i], scores[i])]
#         used.add(i)

#         for j, embedding_j in enumerate(embeddings):
#             if j in used or i == j:
#                 continue

#             similarity = cosine_similarity([embedding_i], [embedding_j])[0][0]
#             # Adjust similarity threshold based on the score
#             x = (high_probability_bonus * min(scores[i], scores[j]) if min(scores[i], scores[j]) > high_probability_bonus else 0)
            
#             adjusted_threshold = similarity_threshold + x
            
#             if similarity > adjusted_threshold:
#                 current_group.append((sentences[j], scores[j]))
#                 used.add(j)

#         # Sort the current group by the probability score in descending order
#         current_group_sorted = sorted(current_group, key=lambda x: x[1], reverse=True)  # Corrected to use x[1] directly

#         # Unpack sentences and scores from the sorted group
#         grouped_sentences, grouped_scores = zip(*current_group_sorted)
#         main_sentence = grouped_sentences[0]
#         similar_sentences = grouped_sentences[1:]

#         group = (main_sentence, grouped_scores[0]), list(zip(similar_sentences, grouped_scores[1:]))
#         groups.append(group)
     
#     mains = [(f[0][0]) for f in groups]


#     # print('MAINS:')
#     # print(mains)

#     # Filter list1 based on the presence of the first item in set_list2
#     # print("SENT W SCORES:")
#     sentences_with_scores = [element for element in sentences_with_scores if element[0] in mains]  
#     # Optionally sort groups by the main sentence's score
#     groups_sorted_by_main_score = sorted(groups, key=lambda g: g[1], reverse=True)

#     # print(sentences_with_scores)
#     # if len(groups_sorted_by_main_score) > 20:  # Limit to top X groups based on your criteria
#     #     groups_sorted_by_main_score = groups_sorted_by_main_score[:30]

#     # just_mains = [(f[0]) for f in groups_sorted_by_main_score]
#     # print(just_mains)

#     return sentences_with_scores #just_mains#groups_sorted_by_main_score


# async def analyze_text_pipeline_pP(sentences):
#     filtered_sentences = filter_sentences_above_threshold(sentences, model, vectorizer)
#     print("BOOM BOOM")
#     return filtered_sentences

# async def analyze_text_pipeline_Terms(sentences):
#     filtered_sentences = filter_sentences_above_threshold(sentences, model2, vectorizer2)
#     flagged_sentences = process_sentences_with_bert(filtered_sentences, tokenizer, bert_model2, device)
#     print("FLAGGED SENTENCES")
#     print(len(flagged_sentences))

#     grouped_sentences = await group_sentences_based_on_embeddings(flagged_sentences, get_text_embeddings_from_list, similarity_threshold=0.81)
    
#     print(len(grouped_sentences))
#     # just_mains = [(f[0],f[0],f[1]) for f in just_mains]
#     # for i in grouped_sentences:
#     #     print(i)
#     return grouped_sentences


def process_privacy_policy(sentences):
    flags = process_sentences_with_bert(sentences, tokenizer, bert_model, device)
    return flags

def process_Terms(sentences):
    flags = process_sentences_with_bert(sentences, tokenizer, bert_model2, device)
    return flags
# Example usage
# if __name__ == "__main__":
#     with open("test_policy_2.txt", 'r', encoding='utf-8') as file:
#         sentences = [line.strip() for line in file.readlines()]
#     analyzed_groups = analyze_text_pipeline(sentences)
#     i = 0
#     for group in analyzed_groups:
#         i +=1
#         print(f"Group: {i}")
#         for sentence in group:
#             print(sentence[0])
