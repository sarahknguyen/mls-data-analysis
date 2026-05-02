# %% [markdown] 
# ## Week 1 – Monthly Dataset Aggregation
# Individual monthly files are combined into unified datasets that span multiple months, enabling trend
# analysis over time.

# %% [markdown]
# ### Objective
# Load and concatenate all monthly MLS files from January 2024 through the most recently completed
# calendar month into analysis-ready combined datasets.
# 
# ### Outputs
# - Combined sold transactions dataset
# - Combined listing data dataset
# 
# ### Skills Learned
# - Multi-file dataset management
# - Data aggregation with Pandas
# - Preparing time-series datasets for analysis
# 

# %%
# This line just hides warning messages so we can focus on the results—but we only use it when we understand what those warnings mean.
import warnings
warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data_path = Path("/Users/sarahbarah3/Desktop/crmls")
save_path = Path("/Users/sarahbarah3/Desktop/crmls/combined")

# %%
def read_csv_with_fallback(filepath):
    """Read CSV with UTF-8, fallback to cp1252 if needed."""
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        return df, "utf-8"
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="cp1252")
        return df, "cp1252"

# %% [markdown]
# ### Define lists

# %%
listing_files = list(data_path.glob("CRMLSListing*.csv"))
sold_files = list(data_path.glob("CRMLSSold*.csv"))

# %%
print(list(data_path.glob("*.csv")))

# %% [markdown]
# ### Read and Concateante

# %%
full_listing = []
encoding_log = []
row_counts = []

for filepath in listing_files:
    df, encoding_used = read_csv_with_fallback(filepath)

    full_listing.append(df)
    encoding_log.append((filepath, encoding_used))
    row_counts.append(len(df))

# Safety check before concat
if full_listing:
    listing_combined = pd.concat(full_listing, ignore_index=True)
    print("Combined listing shape:", listing_combined.shape)
else:
    print("No listing files were loaded.")

# %%
full_sold = []
encoding_log_sold = []
row_counts_sold = []

for filepath in sold_files:
    df, encoding_used = read_csv_with_fallback(filepath)

    full_sold.append(df)
    encoding_log_sold.append((filepath, encoding_used))
    row_counts_sold.append(len(df))

if full_sold:
    sold_combined = pd.concat(full_sold, ignore_index=True)
    print("Combined sold shape:", sold_combined.shape)
else:
    print("No sold files were loaded.")

# %%
listing_combined.to_csv(save_path / "listing_combined.csv", index=False)
sold_combined.to_csv(save_path / "sold_combined.csv", index=False)

# %% [markdown]
# ### View the first and last 5 rows of the listing dataset.

# %%
listing_combined.head()

# %%
listing_combined.tail()

# %% [markdown]
# ### View the first and last 5 rows of the sold dataset.

# %%
sold_combined.head()

# %%
sold_combined.tail()

# %% [markdown]
# ### Overview of the dataset shape (Listing)

# %%
# checking the shape of the data
print(f"There are {sold_combined.shape[0]} rows and {sold_combined.shape[1]} columns in combined listing data.")  # f-string

# %% [markdown]
# ### Overview of the dataset shape (Sold)

# %%
# checking the shape of the data
print(f"There are {sold_combined.shape[0]} rows and {sold_combined.shape[1]} columns in Feb data.")  # f-string

# %% [markdown]
# ### View Listing dataset columns

# %%
listing_combined.info()

# %% [markdown]
# ### Removing Duplicates (Listing)

# %% [markdown]
# ### View Sold dataset columns

# %%
sold_combined.info()

# %% [markdown]
# ### Removing Duplicates

# %%
# Removing Duplicates (Listing)
# alphabetical order, check for duplicates
listing_sorted = listing_combined[sorted(listing_combined.columns)]

# Drop duplicate rows/columns
# Step 1: remove exact duplicates
listing2 = listing_sorted.loc[:, ~listing_sorted.T.duplicated()]

# Step 2: remove .1 columns (extra cleanup)
listing2 = listing_sorted.loc[:, ~listing_sorted.columns.str.endswith('.1')]

listing2.info()

# %%
# Removing Duplicates (Sold)
# alphabetical order, check for duplicates
sold_sorted = sold_combined[sorted(sold_combined.columns)]

# Drop duplicate columns
# Step 1: remove exact duplicates
sold2 = sold_sorted.loc[:, ~sold_sorted.T.duplicated()]

# Step 2: remove .1 columns (extra cleanup)
sold2 = sold2.loc[:, ~sold2.columns.str.endswith('.1')]

# Final check
sold2.info()

# %% [markdown]
# ### View column counts (Listing)

# %%
listing_combined.PropertyType.value_counts()

# %% [markdown]
# ### View column counts (Sold)

# %%
sold_combined.PropertyType.value_counts()

# %% [markdown]
# ### Filter PropertyType to Residential only
# We’re filtering the dataset to only include residential properties so we can analyze the deals that actually matter to us.

# %% [markdown]
# ### Create new datasets

# %%
listing_residential_df = listing_combined[listing_combined["PropertyType"] == "Residential"]
sold_residential_df = sold_combined[sold_combined["PropertyType"] == "Residential"]

# %% [markdown]
# ### Show row and column counts after the Residential filter

# %%
# checking the shape of the data
print(f"There are {listing_residential_df.shape[0]} rows and {listing_residential_df.shape[1]} columns in listing data for residential only.")  # f-string
print(f"There are {sold_residential_df.shape[0]} rows and {sold_residential_df.shape[1]} columns in sold data for residential only.")  # f-string

# %% [markdown]
# ### Save new datasets to CSVs (residential only)

# %%
listing_residential_df.to_csv(save_path / "listings_residential_properties.csv", index=False)
sold_residential_df.to_csv(save_path / "sold_residential_properties.csv", index=False)

# %% [markdown]
# ## Week 2-3 – Data Structuring and Validation
# Before analytics begins, the dataset must be inspected and filtered to ensure only relevant residential
# property records are used.
# 

# %% [markdown]
# ### TO DO: 
# - Separate market analysis fields from metadata fields
# - Decide which columns to drop vs. retain (keep core fields even if partially missing)
# - Numeric Distribution Review: Analyze the distribution of key numeric fields: ClosePrice, ListPrice, OriginalListPrice, LivingArea,
# LotSizeAcres, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, and YearBuilt. For each field,
# generate histograms, boxplots, and percentile summaries, and identify extreme outliers for later handling.
# -  Produce a numeric distribution summary (min, max, mean, median, percentiles) for ClosePrice, LivingArea, and
# DaysOnMarket. 
# - Save the filtered dataset as a new CSV.

# %% [markdown]
# ### Listing

# %%
listing_residential_df.shape

# %%
listing_residential_df.info()

# %%
# Null count + percentage summary
null_summary_listing = pd.DataFrame({
    "Null Count": listing_residential_df.isna().sum(),
    "Missing %": listing_residential_df.isna().mean() * 100
})

# Sort by highest missing %
null_summary_listing = null_summary_listing.sort_values(by="Missing %", ascending=False)

print(null_summary_listing.head(20))

# %%
# Identify high-missing columns above 90%
high_missing_90 = (
    listing_residential_df.isna()
    .mean()
    .mul(100)
    .loc[lambda x: x > 90]
    .sort_values(ascending=False)
)

print(high_missing_90)


# %% [markdown]
# ### Sold

# %%
sold_residential_df.shape

# %%
sold_residential_df.info()

# %%
# Null count + percentage summary
null_summary_sold = pd.DataFrame({
    "Null Count": sold_residential_df.isna().sum(),
    "Missing %": sold_residential_df.isna().mean() * 100
})

# Sort by highest missing %
null_summary_sold = null_summary_sold.sort_values(by="Missing %", ascending=False)

print(null_summary_sold)

# %%
# Identify high-missing columns above 90%
high_missing_90_sold = (
    sold_residential_df.isna()
    .mean()
    .mul(100)
    .loc[lambda x: x > 90]
    .sort_values(ascending=False)
)

print(high_missing_90_sold)


# %% [markdown]
# 

# %% [markdown]
# ### Suggested Intern Questions

# %%
# What is the Residential vs. other property type share?
# Need to used orginal listing_combined to get the full picture of all property types, not just residential
property_summary = (
    listing_combined['PropertyType']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
print(property_summary)

# %%
# What are the median and average close prices?
median_price = round(sold_residential_df['ClosePrice'].median(), 2)
mean_price = round(sold_residential_df['ClosePrice'].mean(), 2)

print("Median Close Price:", median_price)
print("Average Close Price:", mean_price)

# %%
# What does the Days on Market distribution look like?

sold_residential_df['DaysOnMarket'].plot(kind='hist', bins=50)
plt.title("Days on Market Distribution")
plt.xlabel("Days on Market")
plt.ylabel("Frequency")
plt.show()

# %%
sold_residential_df['DaysOnMarket'].describe()

# %%
# Removing outliers by capping at 200 days on market to see the distribution more clearly
sold_residential_df['DaysOnMarket'].clip(upper=200).plot(kind='hist', bins=50)
plt.title("Days on Market Distribution (Capped at 200)")
plt.show()

# %% [markdown]
# The distribution of Days on Market is right-skewed, with most properties selling within a relatively short time frame. The median is lower than the mean, indicating the presence of outliers where some properties remain on the market significantly longer.
# - Most homes sell quickly
# - A few take much longer thus a skew
# - The market is not evenly distributed

# %%
# What percentage of homes sold above vs. below list price?
sdf = sold_residential_df

sdf['Above_List'] = sdf['ClosePrice'] > sdf['ListPrice']

# calculate the percentage of homes sold above vs. below list price
price_comparison = sdf['Above_List'].value_counts(normalize=True) * 100

print(price_comparison)

# ~41% of homes sold above list price, while ~59% sold at or below list price.

# %%
# Are there any apparent date consistency issues (e.g., close date before listing date)?

# We can check for date consistency by comparing the 'CloseDate' and 'ListDate' columns. If there are any rows where 'CloseDate' is before 'ListDate', that would indicate a potential issue.
sdf['CloseDate'] = pd.to_datetime(sdf['CloseDate'], errors='coerce')
sdf['ListingContractDate'] = pd.to_datetime(sdf['ListingContractDate'], errors='coerce') 

date_issues = sdf[sdf['CloseDate'] < sdf['ListingContractDate']]

print("Number of inconsistent records:", len(date_issues))

# %%
date_issues[['ListingContractDate', 'CloseDate']].head()

# %%
# percentage
date_issue_percentage = (len(date_issues) / len(sdf)) * 100
print(f"Percentage of date consistency issues: {date_issue_percentage:.2f}%")

# %% [markdown]
# A small number of records were identified where the close date precedes the listing date, indicating potential data entry or system errors. These records were flagged for further review or removal. Small % indicates normal data noise.

# %% [markdown]
# 

# %%
# Which counties have the highest median prices?

# group by county, calculate median close price, and sort in descending order
county_median = (
    sdf.groupby('CountyOrParish')['ClosePrice']
    .median()
    .sort_values(ascending=False)
)

print(county_median.head(10))

# %%
# Table format for better readability
county_median_df = county_median.reset_index()
county_median_df.columns = ['County', 'Median Close Price']

print(county_median_df.head(10))

# %%
county_median.head(10).plot(kind='bar')
plt.title("Top 10 Counties by Median Close Price")
plt.ylabel("Median Price")
plt.xticks(rotation=45)
plt.show()

# %%



