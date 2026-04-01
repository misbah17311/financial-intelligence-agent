# Downloads and preprocesses the raw datasets:
# 1. Financial news articles from HuggingFace → ChromaDB + BM25
# 2. Company fundamentals (synthetic but realistic) → DuckDB

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.logger import logger


def _try_load_dataset(name, split="train", **kwargs):
    # helper to load a HF dataset with fallback if it fails
    from datasets import load_dataset
    try:
        ds = load_dataset(name, split=split, **kwargs)
        return ds.to_pandas()
    except Exception as e:
        logger.warning(f"  Could not load {name}: {e}. Skipping.")
        return None


def download_financial_news():
    # pull financial news from HuggingFace (mix of large + small datasets)
    # only falls back to synthetic if downloads fail badly
    from datasets import load_dataset

    logger.info("Downloading financial news datasets from HuggingFace...")

    frames = []

    # --- large datasets (these carry the bulk) ---

    # dataset 1: ashraq/financial-news-articles (~300k articles, we take first 80k)
    # loading via streaming to avoid pulling all 300k into RAM at once
    try:
        logger.info("  Loading ashraq/financial-news-articles (streaming, capped at 80k)...")
        ds = load_dataset("ashraq/financial-news-articles", split="train", streaming=True)
        rows = []
        for i, item in enumerate(ds):
            if i >= 80_000:
                break
            title = item.get("title") or ""
            body = item.get("text") or ""
            rows.append({
                "text": f"{title}. {body}" if title else body,
                "source": "financial_news_articles",
                "date": "2023-01-01",
            })
        df = pd.DataFrame(rows)
        frames.append(df)
        logger.info(f"  financial_news_articles: {len(df)} rows")
        del rows  # free memory
    except Exception as e:
        logger.warning(f"  Could not load ashraq/financial-news-articles: {e}")

    # dataset 2: oliverwang15/news_with_gpt_instructions (~40k news items)
    try:
        logger.info("  Loading oliverwang15/news_with_gpt_instructions...")
        ds = load_dataset("oliverwang15/news_with_gpt_instructions", split="train")
        df = ds.to_pandas()
        df = df.rename(columns={"news": "text"})[["text"]].copy()
        df["source"] = "news_with_gpt"
        df["date"] = "2023-06-01"
        frames.append(df)
        logger.info(f"  news_with_gpt: {len(df)} rows")
    except Exception as e:
        logger.warning(f"  Could not load news_with_gpt: {e}")

    # --- smaller datasets (add variety in phrasing and topics) ---

    # dataset 3: zeroshot/twitter-financial-news-sentiment (~12k)
    df = _try_load_dataset("zeroshot/twitter-financial-news-sentiment")
    if df is not None and "text" in df.columns:
        df = df[["text"]].copy()
        df["source"] = "twitter_financial"
        df["date"] = "2022-06-01"
        frames.append(df)
        logger.info(f"  twitter_financial_news: {len(df)} rows")

    # dataset 4: nickmuchi/financial-classification (~4.5k)
    df = _try_load_dataset("nickmuchi/financial-classification")
    if df is not None:
        text_col = next((c for c in ["text", "sentence", "Sentence"] if c in df.columns), None)
        if text_col:
            df = df.rename(columns={text_col: "text"})[["text"]].copy()
            df["source"] = "financial_classification"
            df["date"] = "2022-01-01"
            frames.append(df)
            logger.info(f"  financial_classification: {len(df)} rows")

    # combine everything
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        del frames  # free memory
    else:
        combined = pd.DataFrame(columns=["text", "source", "date"])

    # basic cleaning first to reduce size before anything else
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 20]  # drop junk
    combined = combined.drop_duplicates(subset=["text"])

    # truncate articles to ~1000 chars — keeps the key info, massively cuts chunk count
    combined["text"] = combined["text"].str[:1000]

    total_real = len(combined)
    logger.info(f"  Total real data rows after cleaning: {total_real}")

    # cap at 110k to stay within memory limits
    if total_real > 110_000:
        combined = combined.sample(n=110_000, random_state=42).reset_index(drop=True)
        logger.info(f"  Capped to {len(combined)} rows to stay within memory limits")

    # only generate synthetic if we somehow got less than 100k real rows
    target_rows = 100_000
    if total_real < target_rows:
        shortfall = target_rows - len(combined)
        logger.info(f"  Generating {shortfall} synthetic articles to reach {target_rows}...")
        synthetic = _generate_synthetic_news(shortfall)
        combined = pd.concat([combined, synthetic], ignore_index=True)

    out_path = PROCESSED_DATA_DIR / "financial_news.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(combined)} news articles to {out_path}")
    return combined


def _generate_synthetic_news(n_rows: int) -> pd.DataFrame:
    # generate template-based financial news snippets with randomized
    # companies, numbers, and phrasing for meaningful embedding text
    import numpy as np
    np.random.seed(42)

    companies = [
        "Apple", "Microsoft", "Google", "Amazon", "Meta", "NVIDIA", "Tesla",
        "Intel", "AMD", "Salesforce", "JPMorgan Chase", "Goldman Sachs",
        "Bank of America", "Wells Fargo", "Morgan Stanley", "Johnson & Johnson",
        "Pfizer", "UnitedHealth", "AbbVie", "Eli Lilly", "ExxonMobil", "Chevron",
        "Walmart", "Procter & Gamble", "Coca-Cola", "Nike", "Netflix",
        "Visa", "Mastercard", "Berkshire Hathaway", "Adobe", "Oracle",
        "PayPal", "Broadcom", "Costco", "PepsiCo", "Merck", "Thermo Fisher",
        "McDonald's", "Starbucks", "Home Depot", "ConocoPhillips", "Target",
        "Moderna", "CrowdStrike", "Datadog", "Snowflake", "Palantir", "Uber",
    ]
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
    company_sector = {}
    for c in companies:
        if c in ["Apple", "Microsoft", "Google", "Amazon", "Meta", "NVIDIA", "Tesla", "Intel", "AMD", "Salesforce", "Adobe", "Oracle", "PayPal", "Broadcom", "Netflix", "CrowdStrike", "Datadog", "Snowflake", "Palantir", "Uber"]:
            company_sector[c] = "Technology"
        elif c in ["Johnson & Johnson", "Pfizer", "UnitedHealth", "AbbVie", "Eli Lilly", "Merck", "Thermo Fisher", "Moderna"]:
            company_sector[c] = "Healthcare"
        elif c in ["JPMorgan Chase", "Goldman Sachs", "Bank of America", "Wells Fargo", "Morgan Stanley", "Visa", "Mastercard", "Berkshire Hathaway"]:
            company_sector[c] = "Finance"
        elif c in ["ExxonMobil", "Chevron", "ConocoPhillips"]:
            company_sector[c] = "Energy"
        else:
            company_sector[c] = "Consumer"

    templates = [
        "{company} reported quarterly revenue of ${amount}B, {change}% {direction} year-over-year, {beat_miss} analyst expectations.",
        "Analysts at {bank} upgraded {company} to {rating}, citing strong {metric} growth and improving margins in the {sector} sector.",
        "{company} shares {move} {pct}% after the company announced {event}. The {sector} sector {reaction} on the news.",
        "The {sector} sector faces headwinds as {challenge}. {company} and peers are expected to see {impact} in coming quarters.",
        "{company}'s CEO stated that {initiative} would drive growth in {year}, with expected revenue impact of ${amount}B.",
        "Wall Street is divided on {company}: bulls point to {positive}, while bears worry about {negative}.",
        "Q{quarter} earnings season shows {sector} companies {trend}. {company} led the pack with {metric} of ${amount}B.",
        "{bank} analysts forecast {company} will achieve ${amount}B in revenue by {year}, driven by {driver}.",
        "Investor sentiment toward {company} turned {sentiment} following {catalyst}. Trading volume spiked {pct}%.",
        "{company} announced a ${amount}B {action}, signaling confidence in its {sector} market position.",
        "Market analysts expect the {sector} sector to {outlook} in {year} as {macro_factor} continues to shape the landscape.",
        "{company} expanded its {product} division, investing ${amount}B in {area}. Competitors like {comp2} are watching closely.",
        "A recent survey of fund managers shows {pct}% are overweight on {sector} stocks, with {company} as the top pick.",
        "Bond yields and {sector} stocks moved in opposite directions as {company} reported mixed results for Q{quarter} {year}.",
        "Regulatory concerns weigh on {company} as {regulator} scrutinizes {issue}. The stock dropped {pct}% on the news.",
        "{company}'s profit margins improved to {margin}% in {year}, up from {prev_margin}% the prior year, driven by {reason}.",
        "Supply chain improvements helped {company} reduce costs by ${amount}M in Q{quarter}, boosting net income expectations.",
        "The {sector} sector saw M&A activity surge with {company} acquiring a smaller rival for ${amount}B.",
        "{company} declared a dividend increase of {pct}%, reflecting strong cash flow generation in the {sector} space.",
        "Institutional investors increased holdings in {company} by {pct}% last quarter, according to SEC filings.",
    ]

    banks = ["Goldman Sachs", "Morgan Stanley", "JPMorgan", "Bank of America", "Citigroup", "Barclays", "UBS", "Credit Suisse", "Deutsche Bank"]
    ratings = ["Buy", "Overweight", "Outperform", "Strong Buy", "Hold"]
    events = ["a major restructuring plan", "a new product launch", "a strategic acquisition", "better-than-expected earnings", "a stock buyback program", "expansion into AI services", "a partnership with a tech giant"]
    challenges = ["rising interest rates pressure valuations", "inflation erodes consumer spending", "regulatory uncertainty increases", "supply chain disruptions persist", "competition intensifies from new entrants"]
    positives = ["strong AI revenue growth", "expanding market share", "improving profit margins", "record cash flow generation", "successful product launches"]
    negatives = ["slowing user growth", "margin compression", "regulatory headwinds", "increased competition", "high valuation multiples"]
    sentiments = ["bullish", "cautious", "bearish", "mixed", "optimistic"]
    catalysts = ["strong earnings results", "a surprise CEO change", "new product announcement", "a major contract win", "downbeat guidance for next quarter"]
    actions = ["share buyback", "acquisition", "capital investment", "debt refinancing", "strategic investment"]
    drivers = ["AI adoption", "cloud migration", "international expansion", "cost optimization", "new product segments"]
    areas = ["artificial intelligence", "cloud computing", "cybersecurity", "autonomous vehicles", "biotechnology", "renewable energy"]
    regulators = ["the SEC", "the FTC", "the DOJ", "European regulators", "China's market authority"]
    issues = ["market practices", "data privacy", "antitrust concerns", "accounting practices", "executive compensation"]
    products = ["cloud", "AI", "enterprise", "consumer", "healthcare IT", "fintech", "payments"]
    reasons = ["operational efficiency", "AI-driven automation", "scale advantages", "pricing power", "cost restructuring"]
    macro_factors = ["Federal Reserve policy", "global trade tensions", "AI investment boom", "consumer spending trends", "energy price volatility"]
    outlooks = ["outperform", "underperform", "rally", "face consolidation", "see mixed results"]
    trends = ["beating estimates broadly", "showing margin compression", "reporting strong top-line growth", "missing on guidance"]
    metrics = ["revenue", "EBITDA", "net income", "free cash flow", "operating profit"]
    years = ["2020", "2021", "2022", "2023", "2024"]

    rows = []
    for i in range(n_rows):
        template = templates[i % len(templates)]
        company = np.random.choice(companies)
        comp2 = np.random.choice([c for c in companies if c != company])
        sector = company_sector[company]

        text = template.format(
            company=company, comp2=comp2, sector=sector,
            amount=round(np.random.uniform(0.5, 150), 1),
            change=round(np.random.uniform(1, 35), 1),
            direction=np.random.choice(["higher", "lower"]),
            beat_miss=np.random.choice(["beating", "missing", "meeting"]),
            bank=np.random.choice(banks),
            rating=np.random.choice(ratings),
            metric=np.random.choice(metrics),
            move=np.random.choice(["surged", "dropped", "climbed", "fell", "rose"]),
            pct=round(np.random.uniform(1, 15), 1),
            event=np.random.choice(events),
            reaction=np.random.choice(["rallied", "pulled back", "held steady"]),
            challenge=np.random.choice(challenges),
            impact=np.random.choice(["revenue pressure", "margin expansion", "mixed results"]),
            initiative=np.random.choice(["AI integration", "cloud expansion", "cost restructuring", "market expansion"]),
            year=np.random.choice(years),
            positive=np.random.choice(positives),
            negative=np.random.choice(negatives),
            sentiment=np.random.choice(sentiments),
            catalyst=np.random.choice(catalysts),
            action=np.random.choice(actions),
            driver=np.random.choice(drivers),
            area=np.random.choice(areas),
            regulator=np.random.choice(regulators),
            issue=np.random.choice(issues),
            product=np.random.choice(products),
            reason=np.random.choice(reasons),
            macro_factor=np.random.choice(macro_factors),
            outlook=np.random.choice(outlooks),
            trend=np.random.choice(trends),
            quarter=np.random.choice([1, 2, 3, 4]),
            margin=round(np.random.uniform(8, 35), 1),
            prev_margin=round(np.random.uniform(5, 30), 1),
        )

        month = np.random.randint(1, 13)
        year = np.random.choice([2020, 2021, 2022, 2023, 2024])
        rows.append({
            "text": text,
            "source": "synthetic_financial_news",
            "date": f"{year}-{month:02d}-01",
        })

    return pd.DataFrame(rows)


def generate_company_fundamentals():
    # generate synthetic but realistic company fundamentals
    # covers 105 companies across 5 sectors, 2020–2024 quarterly
    import numpy as np

    logger.info("Generating company fundamentals dataset...")

    np.random.seed(42)

    sectors = {
        "Technology": {
            "companies": [
                "Apple", "Microsoft", "Google", "Amazon", "Meta",
                "NVIDIA", "Tesla", "Intel", "AMD", "Salesforce",
                "Adobe", "Oracle", "IBM", "Netflix", "Cisco",
                "PayPal", "Qualcomm", "Broadcom", "ServiceNow", "Intuit",
                "Uber", "Snap", "Pinterest", "Dropbox", "Zoom",
                "CrowdStrike", "Datadog", "Snowflake", "Palantir", "Twilio",
            ],
            "tickers": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                "NVDA", "TSLA", "INTC", "AMD", "CRM",
                "ADBE", "ORCL", "IBM", "NFLX", "CSCO",
                "PYPL", "QCOM", "AVGO", "NOW", "INTU",
                "UBER", "SNAP", "PINS", "DBX", "ZM",
                "CRWD", "DDOG", "SNOW", "PLTR", "TWLO",
            ],
            "revenue_range": (5000, 120000),
            "growth_rate": 0.08,
        },
        "Healthcare": {
            "companies": [
                "Johnson & Johnson", "UnitedHealth", "Pfizer", "AbbVie", "Eli Lilly",
                "Merck", "Thermo Fisher", "Abbott Labs", "Danaher", "Amgen",
                "Gilead Sciences", "Regeneron", "Vertex", "Moderna", "BioNTech",
                "Humana", "Cigna", "CVS Health", "Anthem", "HCA Healthcare",
                "Medtronic", "Boston Scientific", "Edwards Lifesciences", "Stryker", "Baxter",
            ],
            "tickers": [
                "JNJ", "UNH", "PFE", "ABBV", "LLY",
                "MRK", "TMO", "ABT", "DHR", "AMGN",
                "GILD", "REGN", "VRTX", "MRNA", "BNTX",
                "HUM", "CI", "CVS", "ANTM", "HCA",
                "MDT", "BSX", "EW", "SYK", "BAX",
            ],
            "revenue_range": (3000, 90000),
            "growth_rate": -0.02,
        },
        "Finance": {
            "companies": [
                "JPMorgan Chase", "Bank of America", "Wells Fargo", "Goldman Sachs", "Morgan Stanley",
                "Citigroup", "US Bancorp", "Charles Schwab", "BlackRock", "State Street",
                "Capital One", "American Express", "Visa", "Mastercard", "Discover",
                "Berkshire Hathaway", "MetLife", "Prudential", "AIG", "Aflac",
            ],
            "tickers": [
                "JPM", "BAC", "WFC", "GS", "MS",
                "C", "USB", "SCHW", "BLK", "STT",
                "COF", "AXP", "V", "MA", "DFS",
                "BRK.B", "MET", "PRU", "AIG", "AFL",
            ],
            "revenue_range": (8000, 150000),
            "growth_rate": 0.04,
        },
        "Energy": {
            "companies": [
                "ExxonMobil", "Chevron", "ConocoPhillips", "Schlumberger", "EOG Resources",
                "Pioneer Natural", "Devon Energy", "Marathon Petroleum", "Valero", "Phillips 66",
                "Occidental Petroleum", "Hess", "Halliburton", "Baker Hughes", "Kinder Morgan",
            ],
            "tickers": [
                "XOM", "CVX", "COP", "SLB", "EOG",
                "PXD", "DVN", "MPC", "VLO", "PSX",
                "OXY", "HES", "HAL", "BKR", "KMI",
            ],
            "revenue_range": (5000, 100000),
            "growth_rate": 0.03,
        },
        "Consumer": {
            "companies": [
                "Walmart", "Procter & Gamble", "Coca-Cola", "PepsiCo", "Costco",
                "Nike", "McDonald's", "Starbucks", "Home Depot", "Lowe's",
                "Target", "Dollar General", "Colgate-Palmolive", "Estée Lauder", "Mondelez",
            ],
            "tickers": [
                "WMT", "PG", "KO", "PEP", "COST",
                "NKE", "MCD", "SBUX", "HD", "LOW",
                "TGT", "DG", "CL", "EL", "MDLZ",
            ],
            "revenue_range": (4000, 160000),
            "growth_rate": 0.03,
        },
    }

    rows = []
    years = range(2020, 2025)
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    for sector_name, info in sectors.items():
        for i, company in enumerate(info["companies"]):
            # each company gets a base revenue drawn from the sector range
            base_rev = np.random.uniform(*info["revenue_range"])

            for year in years:
                for qi, quarter in enumerate(quarters):
                    # apply yearly growth + seasonal variation + noise
                    year_mult = (1 + info["growth_rate"]) ** (year - 2020)
                    seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * qi / 4)  # Q4 bump etc.
                    noise = np.random.normal(1.0, 0.05)

                    revenue = base_rev * year_mult * seasonal * noise
                    # net income is some fraction of revenue with its own noise
                    margin = np.random.uniform(0.05, 0.25)
                    net_income = revenue * margin * np.random.normal(1.0, 0.1)
                    # total assets ~ 3-8x revenue
                    total_assets = revenue * np.random.uniform(3, 8)
                    # market cap ~ 5-20x net income
                    market_cap = abs(net_income) * np.random.uniform(5, 25)

                    rows.append({
                        "company_name": company,
                        "ticker": info["tickers"][i],
                        "sector": sector_name,
                        "year": year,
                        "quarter": quarter,
                        "revenue_mn": round(revenue, 2),
                        "net_income_mn": round(net_income, 2),
                        "total_assets_mn": round(total_assets, 2),
                        "market_cap_mn": round(market_cap, 2),
                    })

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DATA_DIR / "company_fundamentals.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} company-quarter records to {out_path}")
    return df


def run_ingestion():
    # main entry point — downloads and preps everything
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    news_df = download_financial_news()
    companies_df = generate_company_fundamentals()

    logger.info(
        f"Ingestion complete: {len(news_df)} articles, "
        f"{len(companies_df)} company-quarter records"
    )
    return news_df, companies_df


if __name__ == "__main__":
    run_ingestion()
