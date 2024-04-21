
def check(original_data):
    # check nan
    cleaned_data= original_data.dropna(axis=1)
    print(cleaned_data.keys()==original_data.keys())

    # check gender noise 
    s =0
    for i in cleaned_data["Gender"]:
        if not( i ==1 or i ==2 ):
            s = s+1
    if s == 0 :     
        print("No out gender")

    # check programme noise 
    s =0
    for i in cleaned_data["Programme"]:
        if not( i ==1 or i ==2 or i ==3 or i ==4):
            s = s+1
    if s == 0 :     
        print("No out Programme")
        
    # check Grade noise 
    s =0
    for i in cleaned_data["Grade"]:
        if not(i ==2 or i ==3 ):
            s = s+1
    if s == 0 :     
        print("No out Grade")