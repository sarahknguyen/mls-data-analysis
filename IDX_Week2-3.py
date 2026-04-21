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
listing_final = listing_clean[
    [col for col in pertinent_cols_listing if col in listing_clean.columns]
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
sold_final = sold_clean[
    [col for col in pertinent_cols_sold if col in sold_clean.columns]
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

# %%
# Looping
for col in numeric_cols:
    if col in sold_final.columns:
        print(f"\n===== {col} =====")

        # Drop missing values
        data = sold_final[col].dropna()

        # Percentile summary
        print("Percentiles:")
        print(data.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

        # Histogram
        plt.figure()
        plt.hist(data, bins=50)
        plt.title(f"{col} Histogram")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

        # Boxplot
        plt.figure()
        plt.boxplot(data, vert=False)
        plt.title(f"{col} Boxplot")
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

# %% [markdown]
# ### Numeric Distribution Summary

# %%
summary_cols = ["ClosePrice", "LivingArea", "DaysOnMarket"]

summary = (
    sold_final[summary_cols]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    .T
)
print(summary)

# %% [markdown]
# ### Key Takeaways
# - Heavily right-skewed
# - ClosePrice: has small number of high-value properties significantly inflating the mean relative to the median, as indicated with extreme outliers
# - LivingArea: similar indication, max is unrealistic, unless some commercial property slipped in
# - DaysOnMarket: max days is HIGHLY unrealistic
# 
# This is expected in real estate datasets, where high-value properties and large land sizes create natural variability. These outliers will be carefully evaluated to determine appropriate handling methods.

# %% [markdown]
# ### Suggested Intern Questions

# %%
# What is the Residential vs. other property type share?
# Need to used orginal listing_combined to get the full picture of all property types, not just residential
property_summary = (
    listing_final['PropertyType']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
print(property_summary)

# %%
# What are the median and average close prices?
median_price = round(sold_final['ClosePrice'].median(), 2)
mean_price = round(sold_final['ClosePrice'].mean(), 2)

print("Median Close Price:", median_price)
print("Average Close Price:", mean_price)

# %%
# What does the Days on Market distribution look like?

sold_final['DaysOnMarket'].plot(kind='hist', bins=50)
plt.title("Days on Market Distribution")
plt.xlabel("Days on Market")
plt.ylabel("Frequency")
plt.show()

# %%
sold_final['DaysOnMarket'].describe()

# %%
# Removing outliers by capping at 200 days on market to see the distribution more clearly
sold_final['DaysOnMarket'].clip(upper=200).plot(kind='hist', bins=50)
plt.title("Days on Market Distribution (Capped at 200)")
plt.show()

# %% [markdown]
# The distribution of Days on Market is right-skewed, with most properties selling within a relatively short time frame. The median is lower than the mean, indicating the presence of outliers where some properties remain on the market significantly longer.
# - Most homes sell quickly
# - A few take much longer thus a skew
# - The market is not evenly distributed

# %%
# What percentage of homes sold above vs. below list price?
sdf = sold_final

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

# %% [markdown]
# ### Save the filtered dataset as a new CSV.

# %%
sold_final.to_csv("sold_final.csv", index=False)
listing_final.to_csv("listing_final.csv", index=False)

# %% [markdown]
# ### Tasks

# %%
# Inspect structure
sold_final.columns

# %%
sold_final.head()

# %%

# Check property categories
sold_final['PropertyType'].unique()


# %%
# Filter residential
sold_final = sold_final[sold_final.PropertyType == 'Residential']

# %%
# Validate completeness
sold_final.isnull().sum()

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
sold_final['year_month'] = pd.to_datetime(
    sold_final['CloseDate'], errors='coerce'
).dt.to_period('M')

# LISTING dataset
listing_final['year_month'] = pd.to_datetime(
    listing_final['ListingContractDate'], errors='coerce'
).dt.to_period('M')

# Step 4 – Merge
sold_with_rates = sold_final.merge(
    mortgage_monthly, on='year_month', how='left'
)

listing_with_rates = listing_final.merge(
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



