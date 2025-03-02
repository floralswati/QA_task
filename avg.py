import pandas as pd

# Load the CSV file (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('bert_score_report.csv')

# Assuming the data is ordered as 10 questions per version,
# we create a new column 'version' to mark the version of each question
df['version'] = (df.index // 10) + 1  # This will assign version 1 to questions 1-10, version 2 to questions 11-20, etc.

# Now, group by 'version' and calculate the mean for each metric
average_scores = df.groupby('version').mean()

# Print the average scores for each version
print(average_scores)

# Optionally, save the result to a new CSV file
average_scores.to_csv('bert_avg_report.csv', index=True)

print("Average scores have been calculated and saved to 'average_scores.csv'")
