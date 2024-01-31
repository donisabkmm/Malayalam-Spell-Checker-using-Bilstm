import csv

input_file = '../DataSet/word_dict_B_f.csv'
output_file = '../DataSet/Final_dataset.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    data = [row for row in reader]

modified_data = [[entry.replace('99~', '') for entry in row] for row in data]

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(modified_data)

print("Prefix '1~' removed from CSV file and saved to output.csv")
