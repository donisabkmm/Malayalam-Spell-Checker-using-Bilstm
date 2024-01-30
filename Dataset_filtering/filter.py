
import re
import csv

# Function to extract text between 1~ and `
def extract_text(data):
    return re.findall(r'1~(.*?)`', data)

# Read the .data file and extract text
input_file_path = "../DataSet/word_dict_B.data"  # Replace with the path to your .data file
output_file_path = "../DataSet/word_dict_B.csv"  # Replace with the desired path for the CSV output

data_list = []
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        extracted_texts = extract_text(line)
        if extracted_texts:
            data_list.extend([[text] for text in extracted_texts])

# Write the extracted text to a CSV file
with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data_list)

print("Extraction and conversion completed successfully.")
