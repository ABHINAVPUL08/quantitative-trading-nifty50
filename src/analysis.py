import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -----------------------------
# Helper: safe column resolver
# -----------------------------
def _get_price_column(df):
    """
    Strategy output may contain different price column names.
    We resolve it safely here.
    """
    for col in ["close", "close_spot", "price", "spot_price"]:
        if col in df.columns:
            return col
    raise ValueError(
        f"No price column found. Available columns: {list(df.columns)}"
    )


# ----------------------------------------
# TASK 7 : Outlier Analysis & Advanced Plots
# ----------------------------------------
def perform_outlier_analysis(df):
    """
    Robust outlier analysis.
    Works even if zero trades are generated.
    Always produces output files for submission.
    """

    print("Performing Outlier Analysis (robust version)...")

    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    price_col = _get_price_column(df)

    trades = []
    position = 0
    entry_price = None
    entry_index = None

    # -------------------
    # Reconstruct trades
    # -------------------
    for i in range(len(df)):
        signal = df.iloc[i]["signal"]
        price = df.iloc[i][price_col]

        # Long Entry
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
            entry_index = i

        # Short Entry
        elif signal == -1 and position == 0:
            position = -1
            entry_price = price
            entry_index = i

        # Exit Long
        elif signal == 2 and position == 1:
            pnl = price - entry_price
            trades.append({
                "pnl": pnl,
                "duration": i - entry_index,
                "iv": df.iloc[entry_index].get("iv", np.nan),
                "vega": df.iloc[entry_index].get("vega", np.nan),
                "regime": df.iloc[entry_index].get("regime_label", np.nan),
            })
            position = 0

        # Exit Short
        elif signal == -2 and position == -1:
            pnl = entry_price - price
            trades.append({
                "pnl": pnl,
                "duration": i - entry_index,
                "iv": df.iloc[entry_index].get("iv", np.nan),
                "vega": df.iloc[entry_index].get("vega", np.nan),
                "regime": df.iloc[entry_index].get("regime_label", np.nan),
            })
            position = 0

    # -------------------
    # If NO trades found
    # -------------------
    if len(trades) == 0:
        print("No trades found.")

        # Placeholder CSVs
        pd.DataFrame(columns=["pnl", "duration", "iv", "vega", "regime"]) \
            .to_csv("results/detailed_trades.csv", index=False)

        pd.DataFrame(columns=["pnl", "duration", "iv", "vega", "regime"]) \
            .to_csv("results/outlier_trades.csv", index=False)

        # Placeholder plots
        plt.figure()
        plt.scatter([0], [0])
        plt.title("PnL vs Duration (No Trades)")
        plt.savefig("plots/pnl_duration_scatter.png")
        plt.close()

        plt.figure()
        sns.heatmap(pd.DataFrame([[1]]), annot=True)
        plt.title("Correlation Heatmap (No Trades)")
        plt.savefig("plots/correlation_heatmap.png")
        plt.close()

        plt.figure()
        plt.scatter([0], [0])
        plt.title("IV Distribution (No Trades)")
        plt.savefig("plots/iv_box_plot.png")
        plt.close()

        return

    # -------------------
    # Trade DataFrame
    # -------------------
    trade_df = pd.DataFrame(trades)

    # Z-score
    if trade_df["pnl"].std() == 0:
        trade_df["z_score"] = 0
    else:
        trade_df["z_score"] = (
            trade_df["pnl"] - trade_df["pnl"].mean()
        ) / trade_df["pnl"].std()

    outliers = trade_df[trade_df["z_score"] > 3]

    trade_df.to_csv("results/detailed_trades.csv", index=False)
    outliers.to_csv("results/outlier_trades.csv", index=False)

    print(f"Found {len(outliers)} outlier trades (Z > 3).")

    # -------------------
    # PLOTS
    # -------------------

    # 1. PnL vs Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=trade_df,
        x="duration",
        y="pnl",
        hue="regime",
        palette="coolwarm"
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Trade PnL vs Duration")
    plt.savefig("plots/pnl_duration_scatter.png")
    plt.close()

    # 2. IV Box Plot
    trade_df["Result"] = np.where(trade_df["pnl"] > 0, "Win", "Loss")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=trade_df, x="Result", y="iv")
    plt.title("IV Distribution: Wins vs Losses")
    plt.savefig("plots/iv_box_plot.png")
    plt.close()

    # 3. Correlation Heatmap
    corr = trade_df[["pnl", "duration", "iv", "vega"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Trade Feature Correlation")
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    print("Outlier analysis & plots completed.")
