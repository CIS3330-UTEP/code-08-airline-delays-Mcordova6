import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'

#ARR_DELAY is the column name that should be used as dependent variable (Y).

# Open file
df = pd.read_csv(filename)

# Descriptive Analytics:

# Step 1: View column names for possible independent variables
print(df.columns)

# Step 2: Get arrival delay descriptive statistics based off the airline name
print(df.groupby(df["OP_CARRIER_NAME"])['ARR_DELAY'].describe().round(2))

# Findings:
    # The airlines with the most arrival delays are American (AA), Delta(DL), and United(UA)
    # The airlines with the longest arrival delays are Skywest(OO), American(AA) and Spirit(NK).

# Step 3: Filter df with airlines with most delays
new_df = df.query('OP_CARRIER == "AA" or OP_CARRIER == "DL" or OP_CARRIER == "UA"')

# Step 4: Find linear relationships using scatterplots on the filtered df
new_df.plot.scatter(x="OP_CARRIER", y="ARR_DELAY")
new_df.plot.scatter(x="DEP_DELAY", y="ARR_DELAY")
new_df.plot.scatter(x="WEATHER_DELAY", y="ARR_DELAY")
new_df.plot.scatter(x="CARRIER_DELAY", y="ARR_DELAY")
new_df.plot.scatter(x="NAS_DELAY", y="ARR_DELAY")

# Step 5: Show scatterplots
plt.show()
 
input()

# Predictive Analytics:

# Step 1: Identify dependent variable (y) and independent variables (x)
y = new_df["ARR_DELAY"]
x = new_df[['DEP_DELAY', 'WEATHER_DELAY', 'CARRIER_DELAY', 'NAS_DELAY']]
x = sm.add_constant(x)

# Step 2: Create OLS regression model
model = sm.OLS(y,x).fit()
print(model.summary())

# Step 3: Create regression equation
# 5.2487 + 0.8239() + 0.1747() + 0.1317() + 0.4156() = ?