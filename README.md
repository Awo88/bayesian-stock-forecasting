# Bayesian Stock Return Forecasting

Applies a **Normal-Normal conjugate Bayesian model** to estimate expected daily stock returns using historical price data, then forecasts future price paths via Monte Carlo simulation.

-----

## Research Question

Can Bayesian inference provide better-calibrated estimates of expected stock returns compared to using sample means alone — and how does the prior belief interact with observed data?

-----

## Dataset

**Source:** Yahoo Finance via `quantmod`
**Window:** 365 trading days (prior year)
**Observations:** ~252 complete trading days after cleaning

|Ticker|Company                |
|------|-----------------------|
|NVDA  |NVIDIA Corp            |
|HOOD  |Robinhood Markets      |
|OGN   |Organon & Co           |
|VG    |Vonage Communications  |
|SPY   |S&P 500 ETF (benchmark)|

-----

## Model

**Likelihood:** daily log returns $r_t \sim \text{Normal}(\mu, \sigma^2)$, where $\sigma$ is estimated from data

**Prior:** $\mu \sim \text{Normal}(\mu_0, \sigma^2/\kappa_0)$, with $\mu_0 = 0.04%$/day and $\kappa_0 = 5$ (weakly informative)

**Posterior (closed-form conjugate update):**

$$\mu_n = \frac{\kappa_0 \mu_0 + n\bar{x}}{\kappa_0 + n}, \qquad \kappa_n = \kappa_0 + n$$

The posterior is $\mu \mid \text{data} \sim \text{Normal}(\mu_n,\ \sigma^2/\kappa_n)$.

### Monte Carlo Simulation

For each of 5,000 paths: draw $\mu$ from the posterior, simulate 63 days of returns from $\text{Normal}(\mu, \sigma^2)$, compound to get price relatives.

-----

## Output Files

|File                         |Description                                                |
|-----------------------------|-----------------------------------------------------------|
|`01_return_distributions.png`|Histogram of daily log returns by stock                    |
|`02_cumulative_returns.png`  |Cumulative return paths over observation period            |
|`03_posterior_means.png`     |Posterior mean daily return with 95% credible intervals    |
|`04_monte_carlo_fan.png`     |63-day simulated price paths (10/25/50/75/90th percentiles)|

-----

## Installation & Usage

```r
install.packages(c("quantmod", "ggplot2", "dplyr", "tidyr", "scales"))
source("bayesian_stock_forecasting.R")
```

-----

## Key Parameters

|Parameter      |Default|Description                              |
|---------------|-------|-----------------------------------------|
|`LOOKBACK_DAYS`|365    |Historical window for likelihood         |
|`PRIOR_MU`     |0.0004 |Prior mean daily return (~10% annualized)|
|`PRIOR_KAPPA`  |5      |Prior strength (equivalent observations) |
|`N_SIM`        |5,000  |Monte Carlo paths per stock              |
|`HORIZON`      |63     |Forecast horizon (trading days)          |

-----

## Limitations

- Assumes returns are normally distributed — fat tails not captured
- Constant volatility (σ) assumption; does not account for volatility clustering
- Uses closing prices only; microstructure effects ignored

-----

*Author: Adebola Awokoya — (2025)*
