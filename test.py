input_file_path = 'input_sentences.txt'
output_file_path = 'sorted_sentences.txt'

# Read the sentences from the input file
with open(input_file_path, 'r') as file:
    sentences = file.readlines()

# Strip any leading/trailing whitespace from each sentence
sentences = [sentence.strip() for sentence in sentences]

# Sort the sentences by length in descending order
sorted_sentences = sorted(sentences, key=len, reverse=False)

# Write the sorted sentences to the output file
with open(output_file_path, 'w') as file:
    for sentence in sorted_sentences:
        file.write(sentence + '\n')

print(f"Sorted sentences have been written to {output_file_path}")