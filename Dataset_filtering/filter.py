import csv

malayalam_words = []
with open('../DataSet/word_dict_A.data', 'r', encoding='utf-8') as file:
    for line in file:
        word = line.strip()
        malayalam_words.append(word)

csv_file = 'malayalam_words.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Malayalam Word'])  # Write header
    for word in malayalam_words:
        writer.writerow([word])

print(f"Malayalam words have been exported to '{csv_file}'.")