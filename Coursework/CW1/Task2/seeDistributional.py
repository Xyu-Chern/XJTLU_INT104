# Task 2 Step 1

import pandas as pd
import matplotlib.pyplot as plt
from checkError import check
original_data = pd.read_excel("Coursework/CW_Data.xlsx")

# check error value
check(original_data)

cleaned_data= original_data.drop(['Index','Programme'], axis="columns")
# see all distributional  
cleaned_data.hist()
plt.show()

