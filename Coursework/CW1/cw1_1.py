import pandas as pd
import matplotlib.pyplot as plt

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
# print(original_data)

# # see all distributional
# original_data.hist()

# # show raw data box plot
plt.boxplot(original_data)
plt.title('Box Plot')  
plt.xlabel('Variables') 
plt.ylabel('Values')  
plt.xticks(range(1, len(original_data.columns) + 1), original_data.keys(), rotation=45)  
plt.show()

print(original_data.keys())
# Index(['Index', 'Gender', 'Programme', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5'],dtype='object')

cleaned_data= original_data.drop(['Index'], axis="columns")
plt.boxplot(cleaned_data)
plt.title('Box Plot')  
plt.xlabel('Variables') 
plt.ylabel('Values')  
plt.xticks(range(1, len(cleaned_data.columns) + 1), cleaned_data.keys(), rotation=45)  
plt.show()


