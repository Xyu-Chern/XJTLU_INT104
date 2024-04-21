# Task 1 Step 1

import pandas as pd
import matplotlib.pyplot as plt

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
# print(original_data)

# show raw data box plot
plt.boxplot(original_data)
plt.title('Box Plot')  
plt.xlabel('Variables') 
plt.ylabel('Values')  
plt.xticks(range(1, len(original_data.columns) + 1), original_data.keys(), rotation=25)  
plt.show()




