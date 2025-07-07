import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 1) Load the dataset
# --------------------------------------------------
csv_path = "cost_revenue_dirty.csv"
df = pd.read_csv(csv_path)

# --------------------------------------------------
# 2) Clean currency columns
# --------------------------------------------------
currency_cols = [
    "USD_Production_Budget",
    "USD_Worldwide_Gross",
    "USD_Domestic_Gross",
]

for col in currency_cols:
    df[col] = df[col].replace({r"[$,]": ""}, regex=True).astype(float)

# --------------------------------------------------
# 3) Convert Release_Date
# --------------------------------------------------
df["Release_Date"] = pd.to_datetime(df["Release_Date"], dayfirst=False, errors="coerce")

# --------------------------------------------------
# 4) Overall summary statistics
# --------------------------------------------------
avg_budget = df["USD_Production_Budget"].mean()
avg_worldwide_gross = df["USD_Worldwide_Gross"].mean()
min_worldwide_gross = df["USD_Worldwide_Gross"].min()
min_domestic_gross = df["USD_Domestic_Gross"].min()

print("--------------------------------------------------")
print("DATASET OVERVIEW")
print("--------------------------------------------------")
print("Rows, columns         :", df.shape)
print("Release date range    :", df['Release_Date'].min().date(), "→", df['Release_Date'].max().date())
print(f"Average Production Budget   : ${avg_budget:,.2f}")
print(f"Average Worldwide Gross     : ${avg_worldwide_gross:,.2f}")
print(f"Minimum Worldwide Gross     : ${min_worldwide_gross:,.2f}")
print(f"Minimum Domestic  Gross     : ${min_domestic_gross:,.2f}")

# --------------------------------------------------
# 5) Profit calculation
# --------------------------------------------------
df["Profit"] = df["USD_Worldwide_Gross"] - df["USD_Production_Budget"]

# --------------------------------------------------
# 6) Bottom‑quartile profitability analysis
# --------------------------------------------------
q1 = df["USD_Worldwide_Gross"].quantile(0.25)
bottom_25 = df[df["USD_Worldwide_Gross"] <= q1]
profitable_cnt = (bottom_25["Profit"] > 0).sum()
loss_cnt = len(bottom_25) - profitable_cnt
pct_profitable = profitable_cnt / len(bottom_25) * 100

print("\n--------------------------------------------------")
print("BOTTOM 25 % PROFITABILITY")
print("--------------------------------------------------")
print(f"Movies in bottom 25 % of worldwide gross : {len(bottom_25)}")
print(f"Profitable (Profit > 0)                  : {profitable_cnt}")
print(f"Unprofitable (Profit ≤ 0)                : {loss_cnt}")
print(f"Percentage profitable                    : {pct_profitable:.2f}%")

# --------------------------------------------------
# 7) Profit distribution plot
# --------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(bottom_25["Profit"], bins=30, kde=True)
plt.title("Profit Distribution – Bottom 25 % Movies")
plt.xlabel("Profit ($)")
plt.axvline(0, color="red", linestyle="--")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8) Budget and revenue extremes
# --------------------------------------------------
max_budget_row = df.loc[df["USD_Production_Budget"].idxmax()]
max_gross_row = df.loc[df["USD_Worldwide_Gross"].idxmax()]
min_budget_row = df.loc[df["USD_Production_Budget"].idxmin()]

print("\n--------------------------------------------------")
print("EXTREMES: BUDGET & REVENUE")
print("--------------------------------------------------")
print(f"Highest Production Budget : ${max_budget_row['USD_Production_Budget']:,.2f} → {max_budget_row['Movie_Title']}")
print(f" → Worldwide Gross : ${max_budget_row['USD_Worldwide_Gross']:,.2f}")
print(f"Highest Worldwide Gross : ${max_gross_row['USD_Worldwide_Gross']:,.2f} → {max_gross_row['Movie_Title']}")
print(f" → Budget : ${max_gross_row['USD_Production_Budget']:,.2f}")
print(f"Lowest Production Budget : ${min_budget_row['USD_Production_Budget']:,.2f} → {min_budget_row['Movie_Title']}")
print(f" → Worldwide Gross : ${min_budget_row['USD_Worldwide_Gross']:,.2f}")

# --------------------------------------------------
# 9) Zero revenue films
# --------------------------------------------------
zero_domestic = df[df["USD_Domestic_Gross"] == 0]
zero_worldwide = df[df["USD_Worldwide_Gross"] == 0]

print("\nTop 5 highest-budget films with $0 domestic gross:")
print(zero_domestic.sort_values("USD_Production_Budget", ascending=False).head(5)[["Movie_Title", "USD_Production_Budget"]])

print("\nTop 5 highest-budget films with $0 worldwide gross:")
print(zero_worldwide.sort_values("USD_Production_Budget", ascending=False).head(5)[["Movie_Title", "USD_Production_Budget"]])

# --------------------------------------------------
# 10) International-only releases
# --------------------------------------------------
intl_only = df.query("USD_Worldwide_Gross > 0 and USD_Domestic_Gross == 0")
print("\nInternational-only releases:")
print(intl_only[["Movie_Title", "USD_Production_Budget", "USD_Worldwide_Gross"]].head())

# --------------------------------------------------
# 11) Filter out unreleased films
# --------------------------------------------------
cutoff = pd.Timestamp("2018-05-01")
data_clean = df[df["Release_Date"] <= cutoff].copy()
unreleased = df[df["Release_Date"] > cutoff]
print(f"\nUnreleased films: {len(unreleased)}")

# --------------------------------------------------
# 12) Films that lost money
# --------------------------------------------------
losses = (data_clean["USD_Production_Budget"] > data_clean["USD_Worldwide_Gross"]).sum()
print(f"\nPercentage of films that lost money: {losses / len(data_clean) * 100:.2f}%")

# --------------------------------------------------
# 13) Regression
# --------------------------------------------------
X = data_clean[["USD_Production_Budget"]]
y = data_clean["USD_Worldwide_Gross"]
model = LinearRegression()
model.fit(X, y)

theta_0 = model.intercept_
theta_1 = model.coef_[0]
print("\n--------------------------------------------------")
print("LINEAR REGRESSION: BUDGET → REVENUE")
print("--------------------------------------------------")
print(f"Intercept (θ₀): ${theta_0:,.2f}")
print(f"Slope     (θ₁): {theta_1:.2f} (revenue per $1 budget)")

# Predict revenue for $150M and $350M budgets
for budget in [150_000_000, 350_000_000]:
    pred = model.predict([[budget]])[0]
    print(f"\nEstimated Worldwide Gross for ${budget / 1_000_000:.0f}M budget: ${pred:,.2f}")
