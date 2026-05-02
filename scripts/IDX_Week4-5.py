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

# Turn off scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)

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
# ### Read and Concatenate

# %%
# Load and combine multiple listing datasets with encoding fallback, while tracking file encodings and row counts
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
# Load and combine multiple sold datasets with encoding fallback, while tracking file encodings and row counts
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
print(f"There are {listing_combined.shape[0]} rows and {listing_combined.shape[1]} columns in combined listing data.")  # f-string

# %% [markdown]
# ### Overview of the dataset shape (Sold)

# %%
# checking the shape of the data
print(f"There are {sold_combined.shape[0]} rows and {sold_combined.shape[1]} columns in combined sold data.")  # f-string

# %% [markdown]
# ### View Listing dataset columns

# %%
listing_combined.info()

# %% [markdown]
# ### View Sold dataset columns

# %%
sold_combined.info()

# %% [markdown]
# ### Removing Duplicates

# %%
# Removing Duplicates (Listing)
# Sort columns
listing_clean = listing_combined[sorted(listing_combined.columns)]

# Remove .1 columns
listing_clean = listing_clean.loc[
    :, ~listing_clean.columns.str.endswith(".1")
]

# Remove duplicate column names
listing_clean = listing_clean.loc[
    :, ~listing_clean.columns.duplicated()
]

listing_clean.info()

# %%
# Removing Duplicates (Sold)
# Sort columns
sold_clean = sold_combined[sorted(sold_combined.columns)]

# Remove .1 columns
sold_clean = sold_clean.loc[
    :, ~sold_clean.columns.str.endswith(".1")
]

# Remove duplicate column names
sold_clean = sold_clean.loc[
    :, ~sold_clean.columns.duplicated()
]

# Final check
sold_clean.info()

# %% [markdown]
# ### View column counts (Listing)

# %%
listing_clean.PropertyType.value_counts()

# %% [markdown]
# ### View column counts (Sold)

# %%
sold_clean.PropertyType.value_counts()

# %% [markdown]
# ### Filter PropertyType to Residential only
# We’re filtering the dataset to only include residential properties so we can analyze the deals that actually matter to us.

# %% [markdown]
# ### Create new datasets

# %%
listing_residential_df = listing_clean[listing_clean["PropertyType"] == "Residential"]
sold_residential_df = sold_clean[sold_clean["PropertyType"] == "Residential"]

# %%
# Sanity Check
listing_residential_df["PropertyType"].unique()
sold_residential_df["PropertyType"].unique()

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
# ## Week 2 – Data Structuring and Validation
# Before analytics begins, the dataset must be inspected and filtered to ensure only relevant residential
# property records are used.
# 

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
# Step 1: Calculate missing percentage
missing_listing = (
    listing_residential_df.isna()
    .mean()
    .mul(100)
    .to_frame(name="Missing %")
)

# Step 2: Flag columns >90% missing
missing_listing["Flag >90%"] = missing_listing["Missing %"] > 90

# Step 3: Filter flagged columns (optional)
high_missing_90_listing = missing_listing[missing_listing["Flag >90%"]]

print("Full Missing Summary (Listing):")
print(missing_listing.sort_values(by="Missing %", ascending=False))

print("\nColumns >90% missing:")
print(high_missing_90_listing)


# %% [markdown]
# ### Separate market analysis fields from metadata fields (Listing)

# %%
# Metadata fields
metadata_cols_list = [
    # IDs / system fields
    "ListingKey", "ListingKeyNumeric", "ListingId",

    # Agent info
    "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "ListAgentEmail", "CoListAgentFirstName", "CoListAgentLastName",
    "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerAgentMlsId",
    "CoBuyerAgentFirstName",

    # Office info
    "ListOfficeName", "BuyerOfficeName", "BuyerOfficeAOR", "CoListOfficeName",

    # Compensation
    "BuyerAgencyCompensation", "BuyerAgencyCompensationType",

    # Builder / misc
    "BuilderName", "BusinessType"
]

# Dropping Metadata Fields
listing_residential_df = listing_residential_df.drop(
    columns=metadata_cols_list,
    errors="ignore"
)

# %%
#  Market Analysis Fields
market_fields_listing = [
    # Pricing
    "ListPrice", "OriginalListPrice",

    # Time / activity
    "DaysOnMarket", "ListingContractDate", "ContractStatusChangeDate",

    # Property characteristics
    "LivingArea", "BuildingAreaTotal",
    "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeAcres", "LotSizeSquareFeet",
    "YearBuilt", "Stories", "Levels",
    "GarageSpaces", "ParkingTotal",
    "FireplacesTotal", "FireplaceYN",
    "NewConstructionYN", "AttachedGarageYN",
    "CoveredSpaces", "MainLevelBedrooms",

    # Location
    "City", "CountyOrParish", "PostalCode",
    "StateOrProvince", "Latitude", "Longitude",
    "UnparsedAddress", "SubdivisionName", "MLSAreaMajor",

    # Property classification
    "PropertyType", "PropertySubType", "MlsStatus",

    # School / neighborhood
    "ElementarySchool", "MiddleOrJuniorSchool", "HighSchool",
    "ElementarySchoolDistrict", "MiddleOrJuniorSchoolDistrict", "HighSchoolDistrict"
]

# Drop everything but these
listing_market_df = listing_residential_df[
    [col for col in market_fields_listing if col in listing_residential_df.columns]
]

# %% [markdown]
# ### Retained Columns (Listing)

# %%
# Dropping School / neighborhood fields (for now)
pertinent_cols_listing = [
    # Pricing
    "ListPrice", "OriginalListPrice",

    # Time / activity
    "DaysOnMarket", "ListingContractDate", "ContractStatusChangeDate",

    # Property characteristics
    "LivingArea", "BuildingAreaTotal",
    "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeAcres", "LotSizeSquareFeet",
    "YearBuilt", "Stories", "Levels",
    "GarageSpaces", "ParkingTotal",
    "FireplacesTotal", "FireplaceYN",
    "NewConstructionYN", "AttachedGarageYN",
    "CoveredSpaces", "MainLevelBedrooms",

    # Location (VERY important)
    "City", "CountyOrParish", "PostalCode",
    "StateOrProvince", "Latitude", "Longitude",
    "UnparsedAddress", "SubdivisionName", "MLSAreaMajor",

    # Property classification
    "PropertyType", "PropertySubType", "MlsStatus"
]

# Final Cleaning
listing_final = listing_residential_df[
    [col for col in pertinent_cols_listing if col in listing_residential_df.columns]
]

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
# Step 1: Calculate missing percentage
missing_sold = (
    sold_residential_df.isna()
    .mean()
    .mul(100)
    .to_frame(name="Missing %")
)

# Step 2: Flag columns >90% missing
missing_sold["Flag >90%"] = missing_sold["Missing %"] > 90

# Step 3: Filter flagged columns (optional)
high_missing_90_sold = missing_sold[missing_sold["Flag >90%"]]

print("Full Missing Summary (Sold):")
print(missing_sold.sort_values(by="Missing %", ascending=False))

print("\nColumns >90% missing:")
print(high_missing_90_sold)

# %% [markdown]
# ### Separate market analysis fields from metadata fields (Sold)

# %%
# Metadata Fields
metadata_cols_sold = [
    # IDs / system
    "ListingKey", "ListingKeyNumeric", "ListingId",

    # Agent info
    "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "ListAgentEmail", "CoListAgentFirstName", "CoListAgentLastName",
    "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerAgentMlsId",
    "CoBuyerAgentFirstName",

    # Office info
    "ListOfficeName", "BuyerOfficeName", "BuyerOfficeAOR", "CoListOfficeName",

    # Compensation
    "BuyerAgencyCompensation", "BuyerAgencyCompensationType",

    # Builder / misc
    "BuilderName", "BusinessType"
]

# Dropping Metadata Fields
sold_residential_df = sold_residential_df.drop(
    columns=metadata_cols_sold,
    errors="ignore"
)


# %%
# Market Data Analysis (Sold)
market_fields_sold = [
    # Pricing (MOST IMPORTANT)
    "ClosePrice", "ListPrice", "OriginalListPrice",

    # Time / transaction
    "CloseDate", "PurchaseContractDate",
    "ListingContractDate", "ContractStatusChangeDate",
    "DaysOnMarket",

    # Property characteristics
    "LivingArea", "BuildingAreaTotal",
    "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeAcres", "LotSizeSquareFeet",
    "YearBuilt", "Stories", "Levels",
    "GarageSpaces", "ParkingTotal",
    "FireplacesTotal", "FireplaceYN",
    "NewConstructionYN", "AttachedGarageYN",
    "CoveredSpaces", "MainLevelBedrooms",
    "AboveGradeFinishedArea", "BelowGradeFinishedArea",

    # Location (CRITICAL)
    "City", "CountyOrParish", "PostalCode",
    "StateOrProvince", "Latitude", "Longitude",
    "UnparsedAddress", "SubdivisionName", "MLSAreaMajor",

    # Property classification
    "PropertyType", "PropertySubType", "MlsStatus",

    # School / neighborhood
    "ElementarySchool", "MiddleOrJuniorSchool", "HighSchool",
    "ElementarySchoolDistrict", "MiddleOrJuniorSchoolDistrict", "HighSchoolDistrict"
]

# Keep one these
sold_market_df = sold_residential_df[
    [col for col in market_fields_sold if col in sold_residential_df.columns]
]

# %% [markdown]
# ### Retained Columns (Sold)

# %%
# Dropping School / neighborhood fields (for now)
pertinent_cols_sold = [
    # Pricing (MOST IMPORTANT)
    "ClosePrice", "ListPrice", "OriginalListPrice",

    # Time / transaction
    "CloseDate", "PurchaseContractDate",
    "ListingContractDate", "ContractStatusChangeDate",
    "DaysOnMarket",

    # Property characteristics
    "LivingArea", "BuildingAreaTotal",
    "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeAcres", "LotSizeSquareFeet",
    "YearBuilt", "Stories", "Levels",
    "GarageSpaces", "ParkingTotal",
    "FireplacesTotal", "FireplaceYN",
    "NewConstructionYN", "AttachedGarageYN",
    "CoveredSpaces", "MainLevelBedrooms",
    "AboveGradeFinishedArea", "BelowGradeFinishedArea",

    # Location (CRITICAL)
    "City", "CountyOrParish", "PostalCode",
    "StateOrProvince", "Latitude", "Longitude",
    "UnparsedAddress", "SubdivisionName", "MLSAreaMajor",

    # Property classification
    "PropertyType", "PropertySubType", "MlsStatus"
]
# Final Cleaning
sold_final = sold_residential_df[
    [col for col in pertinent_cols_sold if col in sold_residential_df.columns]
]

# %% [markdown]
# ### Numeric Distribution Review
# Analyze the distribution of key numeric fields: ClosePrice, ListPrice, OriginalListPrice, LivingArea, LotSizeAcres, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, and YearBuilt. For each field, generate histograms, boxplots, and percentile summaries, and identify extreme outliers for later handling.

# %%
numeric_cols = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres",   
    "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt"
]

# %% [markdown]
# ### Number Distribution Summary

# %%
# Percentile Summaries
summary = []

for col in numeric_cols:
    data = sold_final[col].dropna()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower) | (data > upper)]
    
    summary.append({
        "Column": col,
        "Mean": data.mean(),
        "Median": data.median(),
        "Min": data.min(),
        "Max": data.max(),
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Outliers (#)": len(outliers), 
        "Outliers (%)": round(len(outliers)/len(data)*100, 2)
    })

summary_df = pd.DataFrame(summary)
summary_df

# %%
# Boxplots for numeric columns
n = len(numeric_cols)

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)  # adjust grid if needed
    plt.boxplot(sold_final[col].dropna(), vert=False)
    plt.title(col)

plt.tight_layout()
plt.show()

# %%
# Combined histograms in grid
sold_final[numeric_cols].hist(bins=50, figsize=(12, 8))
plt.suptitle("Histograms for Numeric Features")
plt.show()

# %% [markdown]
# ### Identify extreme outliers (IQR Method)

# %%
outliers_dict = {}

for col in numeric_cols:
    if col in sold_final.columns:
        data = sold_final[col].dropna()

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        outliers_dict[col] = len(outliers)

        print(f"{col}: {len(outliers)} outliers")

# %%
# Removed Unrealistic Outliers
# Define thresholds based on domain knowledge and summary statistics
thresholds = {
    "LivingArea": 10_000,      # Remove properties larger than 10,000 sqft
    "LotSizeAcres": 5,         # Remove properties with lot size larger than 5 acres
    "BedroomsTotal": 10,       # Remove properties with more than 10 bedrooms
    "BathroomsTotalInteger": 10, # Remove properties with more than 10 bathrooms
    "DaysOnMarket": 365,       # Remove properties that were on the market for more than 1 year
}

# Will remove ClosePrice outliers later, after we analyze the distribution more closely

# %% [markdown]
# ### Key Takeaways
# - Data is heavily right-skewed → median is a better measure than mean
# - ClosePrice: Influenced by high-value outliers (luxury homes) → inflates mean
# - LivingArea: Contains unrealistic max values → possible data quality issues
# - LotSize: Extreme outlier skewing distribution → may need removal or validation
# - DaysOnMarket: Includes negative and unusually high values → likely data errors
# 
# Overall, these patterns are expected in real estate data, but outliers and invalid values should be handled carefully before analysis.

# %% [markdown]
# ### Suggested Intern Questions

# %%
# Q1. What is the Residential vs. other property type share?
property_summary = (
    listing_final['PropertyType']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
print(property_summary)

# %%
# sanity check to make sure all are residential
listing_final["PropertyType"].value_counts()

# %%
# Q2. What are the median and average close prices?
median_price = round(sold_final['ClosePrice'].median(), 2)
mean_price = round(sold_final['ClosePrice'].mean(), 2)

print("Median Close Price:", median_price)
print("Average Close Price:", mean_price)

# %% [markdown]
# - Average is much higher than median, indicating a right-skewed distribution with some high-end outliers
# - The large gap between mean and median suggests the presence of high-end properties, 
# - Reinforcing the need for outlier analysis or segmentation when modeling price trends.

# %%
# IQR method for outlier detection on ClosePrice
Q1 = sold_final["ClosePrice"].quantile(0.25)
Q3 = sold_final["ClosePrice"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Identify outliers
outliers = sold_final[
    (sold_final["ClosePrice"] < lower_bound) |
    (sold_final["ClosePrice"] > upper_bound)
]

print("Number of outliers:", outliers.shape[0])

# Outliers removed
sold_finalQ2 = sold_final[
    (sold_final["ClosePrice"] >= lower_bound) &
    (sold_final["ClosePrice"] <= upper_bound)
].copy()

# %% [markdown]
# - Anything over 2387500.0 are outliers, best to remove them
# - 32609 homes are considered outliers which is ~7.37% of the dataset
# - Not random and definitely due to luxury homes
# - Extreme outliers were removed

# %%
# Q3. Histogram of Days on Market with mean, median, and peak labels
sold_finalQ3 = sold_finalQ2[
    (sold_finalQ2["DaysOnMarket"] >= 0) &
    (sold_finalQ2["DaysOnMarket"] <= 200)
]

mean = sold_finalQ3["DaysOnMarket"].mean()
median = sold_finalQ3["DaysOnMarket"].median()
peak = sold_finalQ3["DaysOnMarket"].mode()[0]

sold_finalQ3['DaysOnMarket'].plot(kind='hist', bins=50)

plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {round(mean,1)}')
plt.axvline(median, color='green', linestyle='-', linewidth=2, label=f'Median: {median}')
plt.axvline(peak, color='purple', linestyle=':', linewidth=2, label=f'Peak: {peak}')


plt.title("Days on Market Distribution (Capped at 200)")
plt.xlabel("Days on Market")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# %%
sold_finalQ3['DaysOnMarket'].describe()

# %% [markdown]
# The distribution of Days on Market is right-skewed, with most properties selling within a relatively short time frame. The median is lower than the mean, indicating the presence of outliers where some properties remain on the market significantly longer.
# - Most homes sell quickly
# - A few take much longer thus a skew
# - The market is not evenly distributed

# %%
# Q3. What percentage of homes sold above vs. below list price?

sold_finalQ3['Above_List'] = sold_finalQ3['ClosePrice'] > sold_finalQ3['ListPrice']

# calculate the percentage of homes sold above vs. below list price
price_comparison = sold_finalQ3['Above_List'].value_counts(normalize=True) * 100

print(price_comparison)

# ~41% of homes sold above list price, while ~58% sold at or below list price.

# %% [markdown]
# - ~42% of homes sold above list price |||| ~58% sold at or below list price
# - Most homes are selling at or below asking
# - So sellers aren't always getting bidding wars
# - Market is not extremely overheated
# 
# - 41% is still high: indicates a strong demand in certain segments and competitive pockets of market

# %%
# Q4. Are there any apparent date consistency issues (e.g., close date before listing date)?

# We can check for date consistency by comparing the 'CloseDate' and 'ListDate' columns. If there are any rows where 'CloseDate' is before 'ListDate', that would indicate a potential issue.
sold_finalQ3['CloseDate'] = pd.to_datetime(sold_finalQ3['CloseDate'], errors='coerce')
sold_finalQ3['ListingContractDate'] = pd.to_datetime(sold_finalQ3['ListingContractDate'], errors='coerce')

date_issues = sold_finalQ3[sold_finalQ3['CloseDate'] < sold_finalQ3['ListingContractDate']]

print("Number of inconsistent records:", len(date_issues))

# %% [markdown]
# - 46 records that showed as sold before it was listed
# - It is logically impossible and the inconsistency are small enough to say it is safe to remove them

# %%
# remove records with date inconsistencies (if any)
sold_finalQ4 = sold_finalQ3[sold_finalQ3["CloseDate"] >= sold_finalQ3["ListingContractDate"]].copy()

# %%
date_issues[['ListingContractDate', 'CloseDate']].head()

# %%
# percentage (GOOD)
date_issue_percentage = (len(date_issues) / len(sold_finalQ4)) * 100
print(f"Percentage of date consistency issues: {date_issue_percentage:.2f}%")

# %%
# Which counties have the highest median prices?

# group by county, calculate median close price, and sort in descending order
county_median = (
    sold_finalQ4.groupby('CountyOrParish')['ClosePrice']
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

# %% [markdown]
# - Bay area dominates
#     - high demand
#     - tech industry
#     - limited supply
# - Coastal counties are expensive
# - Key takeaway: location heavily impacts price

# %%
# Applying the thresholds to remove outliers from the sold_final dataset
sold_final2 = sold_finalQ4.copy()

for col, max_val in thresholds.items():
    sold_final2 = sold_final2[sold_final2[col] <= max_val]

# %%
# Sanity Check
print("Before:", sold_final.shape)
print("After:", sold_final2.shape)

# %% [markdown]
# ### Save the filtered dataset as a new CSV.

# %%
# Copying final cleaned datasets for export
sold_week2 = sold_finalQ4.copy()
listing_week2 = listing_final.copy()

# Save CSVs
sold_week2.to_csv("sold_week2.csv", index=False)
listing_week2.to_csv("listing_week2.csv", index=False)

# %% [markdown]
# ### Final Key Takeaways
# - Identified right-skewed distributions, especially in ClosePrice, due to high-value properties
# - Compared mean vs. median to confirm the impact of luxury outliers
# - Used the IQR method to detect and quantify outliers across numeric variables
# - Removed unrealistic values using domain-based thresholds (e.g., extreme sizes, negative days)
# - Retained meaningful outliers (e.g., luxury homes) to preserve market insights
# - Created a cleaned dataset for more accurate and reliable analysis

# %% [markdown]
# ### Tasks

# %%
# Inspect structure
sold_week2.columns

# %%
sold_week2.head()

# %%
# Check property categories
sold_week2['PropertyType'].unique()


# %%
# Filter residential
sold_week2 = sold_week2[sold_week2.PropertyType == 'Residential']

# %%
# Validate completeness
sold_week2.isnull().sum()

# %% [markdown]
# ## Week 3 - Mortgage Rate Enrichment
# - Enrich both the combined sold and listings datasets by merging in the national 30-year fixed mortgage rate from the St. Louis Federal Reserve (FRED).

# %%
# Step 1 – Fetch mortgage data
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

# Step 2 – Convert to monthly average
mortgage['year_month'] = mortgage['date'].dt.to_period('M')

mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)

# Step 3 – Create matching key in your datasets

# SOLD dataset
sold_week2['year_month'] = pd.to_datetime(
    sold_week2['CloseDate'], errors='coerce'
).dt.to_period('M')

# LISTING dataset
listing_week2['year_month'] = pd.to_datetime(
    listing_week2['ListingContractDate'], errors='coerce'
).dt.to_period('M')

# Step 4 – Merge
sold_with_rates = sold_week2.merge(
    mortgage_monthly, on='year_month', how='left'
)

listing_with_rates = listing_week2.merge(
    mortgage_monthly, on='year_month', how='left'
)

# Step 5 – Validate
print("Missing sold rates:", sold_with_rates['rate_30yr_fixed'].isnull().sum())
print("Missing listing rates:", listing_with_rates['rate_30yr_fixed'].isnull().sum())

# Preview
print(
    sold_with_rates[
        ['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']
    ].head()
)

# %%
# Save to CSV files

sold_with_rates.to_csv("sold_with_rates.csv", index=False)
listing_with_rates.to_csv("listing_with_rates.csv", index=False)

# %% [markdown]
# ## Weeks 4–5 – Data Cleaning and Preparation
# Raw MLS data contains formatting inconsistencies, missing values, and fields that need transformation before analysis. This phase prepares the dataset for reliable analytics.
# 
# Tasks
# - Convert date fields to datetime format (CloseDate, PurchaseContractDate, ListingContractDate, ContractStatusChangeDate)
# - Remove unnecessary or redundant columns (removed earlier)
# - Handle missing values appropriately
# - Ensure numeric fields are properly typed
# - Remove or flag invalid numeric values: ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0, negative Bedrooms or Bathrooms
# 

# %% [markdown]
# ### Convert to Datetime Format

# %%
sold_with_rates.columns

# %%
listing_with_rates.columns

# %%
# Function to convert date columns with error handling
def convert_date_columns(df, date_cols):
    cols = [col for col in date_cols if col in df.columns]
    missing = [col for col in date_cols if col not in df.columns]
    
    if missing:
        print(f"Missing columns: {missing}")
    
    df[cols] = df[cols].apply(pd.to_datetime, errors='coerce')
    
    return df

# %%
# Define date columns for each dataset (Sold)
date_cols_sold = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

sold_with_rates = convert_date_columns(sold_with_rates, date_cols_sold)

# %%
# Define date columns for each dataset (Listing)
date_cols_listing = [
    "ListingContractDate",
    "ContractStatusChangeDate"
]

listing_with_rates = convert_date_columns(listing_with_rates, date_cols_listing)

# %%
# Verify date conversion
sold_with_rates[date_cols_sold].dtypes
#listing_with_rates[date_cols_listing].dtypes

# %% [markdown]
# ### Remove unnecessary or redundant columns (removed earlier in Week 1)

# %%
# Sold
sold_missing = pd.DataFrame({
    "Missing Count": sold_with_rates.isna().sum()
})

sold_missing["Missing %"] = (
    sold_missing["Missing Count"] / len(sold_with_rates)
).round(2)

# Listing
listing_missing = pd.DataFrame({
    "Missing Count": listing_with_rates.isna().sum()
})

listing_missing["Missing %"] = (
    listing_missing["Missing Count"] / len(listing_with_rates)
).round(2)

# %% [markdown]
# ### Handle missing values appropriately

# %%
from IPython.display import display, HTML

html = f"""
<div style="display: flex; gap: 40px;">
    <div>
        <h4>Sold Dataset</h4>
        {sold_missing[sold_missing["Missing Count"] > 0].to_html()}
    </div>
    <div>
        <h4>Listing Dataset</h4>
        {listing_missing[listing_missing["Missing Count"] > 0].to_html()}
    </div>
</div>
"""

display(HTML(html))

# %%
# Rule-based column removal

datasets = {
    "sold": (sold_with_rates, sold_missing),
    "listing": (listing_with_rates, listing_missing)
}
cleaned_data = {}

for name, (df, missing_summary) in datasets.items():
    print(f"\nProcessing {name} dataset...")


    # 1. Drop columns with >90% missing
    high_missing_cols = missing_summary[
        missing_summary["Missing %"] > 0.90
].index.tolist()

    # 2. Manual drops
    manual_drop_cols = ["SubdivisionName"]

    # 3. Combine
    cols_to_drop = list(set(high_missing_cols + manual_drop_cols))

    # 4. Drop
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")

    # 5. Store result
    cleaned_data[name] = df_clean

    # 6. Output summary
    print("Columns dropped:", cols_to_drop)
    print("Before:", df.shape)
    print("After:", df_clean.shape)

# Decided to keep MainLevelBedrooms despite missingness, as it could be a valuable feature for analysis and modeling.

# %%
# Track Data Loss (SOLD)
before = len(sold_clean)

sold_clean = sold_clean[
    (sold_clean["ClosePrice"] > 0) &
    (sold_clean["LivingArea"] > 0) &
    (sold_clean["DaysOnMarket"] >= 0) &
    (sold_clean["BedroomsTotal"] >= 0) & # Assuming 0 bedrooms is possible (e.g., studio apartments)
    (sold_clean["BathroomsTotalInteger"] >= 0)
]

after = len(sold_clean)
print(f"Sold rows removed: {before - after}")


### SAFER APPROACH JUST IN CASE ONE OF THESE COLUMNS DOESN'T EXIST
required_cols_sold = [
    "ClosePrice", "LivingArea", "DaysOnMarket",
    "BedroomsTotal", "BathroomsTotalInteger"
]

sold_clean = sold_clean.dropna(subset=required_cols_sold)

print(f"Percent removed: {(before - after)/before:.2%}")

# %%
# Track Data Loss (Listing)
before = len(listing_clean)

listing_clean = listing_clean[
    (listing_clean["OriginalListPrice"] > 0) &
    (listing_clean["LivingArea"] > 0) &
    (listing_clean["DaysOnMarket"] >= 0) &
    (listing_clean["BedroomsTotal"] >= 0) &
    (listing_clean["BathroomsTotalInteger"] >= 0)
]

after = len(listing_clean)

required_cols_listing = [
    "OriginalListPrice", "LivingArea", "DaysOnMarket",
    "BedroomsTotal", "BathroomsTotalInteger"
]

listing_clean = listing_clean.dropna(subset=required_cols_listing)

print(f"Listing rows removed: {before - after}")
print(f"Percent removed: {(before - after) / before:.2%}")

# %% [markdown]
# ### Numeric Columns

# %%
# Define Numeric Columns
sold_numeric_cols = [
    "ClosePrice", "LivingArea", "DaysOnMarket",
    "BedroomsTotal", "BathroomsTotalInteger"
]

listing_numeric_cols = [
    "OriginalListPrice", "LivingArea", "DaysOnMarket",
    "BedroomsTotal", "BathroomsTotalInteger"
]

# %%
# Converts numeric columns, handles missing values, and applies validation rules to ensure data quality
def clean_numeric(df, numeric_cols, rules, name="dataset"):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    df = df.dropna(subset=rules.keys())

    for col, condition in rules.items():
        df = df[condition(df[col])]

    after = len(df)
    print(f"{name}: removed {before - after} rows")

    return df


# %%
# Call function (more of a validaion step to ensure numeric columns are clean and meet expected criteria)
sold_rules = {
    "ClosePrice": lambda x: x > 0,
    "LivingArea": lambda x: x > 0,
    "DaysOnMarket": lambda x: x >= 0,
    "BedroomsTotal": lambda x: x >= 0,
    "BathroomsTotalInteger": lambda x: x >= 0  # MAYBE NEVER?
}

listing_rules = {
    "OriginalListPrice": lambda x: x > 0,
    "LivingArea": lambda x: x > 0,
    "DaysOnMarket": lambda x: x >= 0,
    "BedroomsTotal": lambda x: x >= 0,
    "BathroomsTotalInteger": lambda x: x >= 0
}

sold_clean = clean_numeric(sold_clean, sold_numeric_cols, sold_rules, "sold")
listing_clean = clean_numeric(listing_clean, listing_numeric_cols, listing_rules, "listing")

# %% [markdown]
# ### Date Consistency Checks (Flags)

# %%
# Validate date consistency by flagging missing critical dates and invalid event timelines
def add_date_flags(df):
    
    # Missing critical dates
    date_cols = ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
    existing_cols = [col for col in date_cols if col in df.columns]
    
    if existing_cols:
        df["missing_critical_dates_flag"] = df[existing_cols].isna().any(axis=1)

    # Listing after close
    if "ListingContractDate" in df.columns and "CloseDate" in df.columns:
        df["listing_after_close_flag"] = (
            df["ListingContractDate"].notna() &
            df["CloseDate"].notna() &
            (df["ListingContractDate"] > df["CloseDate"])
        )

    # Purchase after close
    if "PurchaseContractDate" in df.columns and "CloseDate" in df.columns:
        df["purchase_after_close_flag"] = (
            df["PurchaseContractDate"].notna() &
            df["CloseDate"].notna() &
            (df["PurchaseContractDate"] > df["CloseDate"])
        )

    # Negative timeline
    if "ListingContractDate" in df.columns and "PurchaseContractDate" in df.columns:
        df["negative_timeline_flag"] = (
            df["ListingContractDate"].notna() &
            df["PurchaseContractDate"].notna() &
            (df["ListingContractDate"] > df["PurchaseContractDate"])
        )

    return df

# %%
sold_clean = add_date_flags(sold_clean)
listing_clean = add_date_flags(listing_clean)

# %%
# Sanity Check
sold_clean.filter(like="flag").sum()

# %%
listing_clean.filter(like="flag").sum()

# %% [markdown]
# ### Numeric Consistency Checks

# %%
# Identify missing and invalid numeric values
# Sold
sold_clean["invalid_price_flag"] = (
    sold_clean["ClosePrice"].isna() | (sold_clean["ClosePrice"] <= 0)
)
sold_clean["invalid_area_flag"] = (
    sold_clean["LivingArea"].isna() | (sold_clean["LivingArea"] <= 0)
)
sold_clean["invalid_dom_flag"] = (
    sold_clean["DaysOnMarket"].isna() | (sold_clean["DaysOnMarket"] < 0)
)

# Listing
listing_clean["invalid_price_flag"] = (
    listing_clean["OriginalListPrice"].isna() | (listing_clean["OriginalListPrice"] <= 0)
)

listing_clean["invalid_area_flag"] = (
    listing_clean["LivingArea"].isna() | (listing_clean["LivingArea"] <= 0)
)

listing_clean["invalid_dom_flag"] = (
    listing_clean["DaysOnMarket"].isna() | (listing_clean["DaysOnMarket"] < 0)
)

# %% [markdown]
# ### Geographic Data Checks

# %%
# Create flags to identify missing, invalid, and out-of-range geographic coordinates
def add_coordinate_flags(df, state_name="California"):
    df["missing_coord_flag"] = (
        df["Latitude"].isna() | df["Longitude"].isna()
    )

    df["zero_coord_flag"] = (
        (df["Latitude"] == 0) | (df["Longitude"] == 0)
    )

    df["positive_longitude_flag"] = (
        df["Longitude"] > 0
    )

    df["out_of_bounds_flag"] = (
        (df["Latitude"] < 32) | (df["Latitude"] > 42) |
        (df["Longitude"] < -125) | (df["Longitude"] > -114)
    )

    coord_flag_cols = [
        "missing_coord_flag",
        "zero_coord_flag",
        "positive_longitude_flag",
        "out_of_bounds_flag"
    ]

    df["any_coord_issue_flag"] = df[coord_flag_cols].any(axis=1)

    print(f"{state_name} coordinate issue counts:")
    print(df[coord_flag_cols].sum())

    return df

# %%
# Call the function for both datasets
sold_clean = add_coordinate_flags(sold_clean)
print("------------------------")
listing_clean = add_coordinate_flags(listing_clean)

# %% [markdown]
# ### Geographic Data Quality Summary
# - A significant number of records have missing coordinates, which limits mapping and spatial analysis
# - A small number of records contain (0,0) coordinates, likely placeholder or default values
# - Some records show positive longitude values, which are invalid for California (should be negative)
# - A subset of records fall outside expected California geographic bounds, indicating potential data entry or geocoding errors
# 
# Overall, most data is usable, but these flagged records may need to be removed or handled depending on analysis goals.
# 
# ### Quick validation summary

# %%
# Define flags
date_flags = [
    "listing_after_close_flag",
    "purchase_after_close_flag",
    "negative_timeline_flag"
]

geo_flags = [
    "missing_coord_flag",
    "zero_coord_flag",
    "positive_longitude_flag",
    "out_of_bounds_flag"
]

numeric_flags = [
    "invalid_price_flag",
    "invalid_area_flag",
    "invalid_dom_flag"
]

all_flags = date_flags + geo_flags + numeric_flags

# %%
# Include percentages in the summary for better context on the prevalence of each issue
def summarize_flags(df, name):
    print(f"\n{name} Dataset Validation Summary")
    print("-" * 50)
    
    total = len(df)
    
    for col in all_flags:
        if col in df.columns:
            count = df[col].sum()
            pct = (count / total * 100) if total > 0 else 0
            print(f"{col}: {count} ({pct:.2f}%) flagged")

# %%
summarize_flags(sold_clean, "Sold")
summarize_flags(listing_clean, "Listing")

# %%
# Create a row-level data quality score by aggregating validation flags
def add_quality_metrics(df):
    flag_cols = [col for col in all_flags if col in df.columns]

    # Total issues per row
    df["total_issue_count"] = df[flag_cols].sum(axis=1)

    # Any issue flag
    df["any_issue_flag"] = df["total_issue_count"] > 0

    # Data quality score (1 = perfect, 0 = worst)
    if len(flag_cols) > 0:
        df["data_quality_score"] = 1 - (df["total_issue_count"] / len(flag_cols))
    else:
        df["data_quality_score"] = 1

    return df

# %%
# Add quality metrics to both datasets
sold_clean = add_quality_metrics(sold_clean)
listing_clean = add_quality_metrics(listing_clean)

# %%
# Add a quick summary function to report the number and percentage of rows with any issues, providing a clear overview of overall data quality
def summarize_group(df, flags, label):
    total = len(df)
    count = df[flags].any(axis=1).sum()
    pct = count / total * 100 if total > 0 else 0
    print(f"{label}: {count} rows ({pct:.2f}%) have issues")

# %%
summarize_group(sold_clean, date_flags, "Date Issues")
summarize_group(sold_clean, geo_flags, "Geo Issues")
summarize_group(sold_clean, numeric_flags, "Numeric Issues")

# %%
summarize_group(listing_clean, date_flags, "Date Issues")
summarize_group(listing_clean, geo_flags, "Geo Issues")
summarize_group(listing_clean, numeric_flags, "Numeric Issues")

# %%
print(sold_clean.dtypes)
print("------------------------")
print(listing_clean.dtypes)

# %% [markdown]
# ### Data Transformation Summary
# - Converted date fields to datetime format to support timeline validation
# - Removed redundant and high-missing columns to improve data usability
# - Standardized numeric fields and ensured proper data types
# - Filtered invalid records (e.g., non-positive price, area, and negative days on market)
# - Created date consistency flags to identify timeline issues across listing, purchase, and close dates (where applicable)
# - Applied geographic validation to flag missing, invalid, and out-of-range coordinates
# 
# Overall, these transformations improved data quality, reduced noise, and ensured both datasets are clean and analysis-ready.

# %% [markdown]
# ### Saved CSV for Week 4-5

# %%
sold_clean.to_csv("sold_week4_5.csv", index=False)
listing_clean.to_csv("listing_week4_5.csv", index=False)


