# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Stock Return Forecasting
# Author: Adebola Awokoya
#
# Applies a Normal-Normal conjugate Bayesian model to estimate expected
# daily stock returns for five equities. Uses historical return data as
# the likelihood and a weakly informative prior based on long-run market
# returns. Forecasts future price paths via Monte Carlo simulation.
#
# Data: Live prices from Yahoo Finance via quantmod
# Install: install.packages(c("quantmod","ggplot2","dplyr","tidyr","scales"))
# ─────────────────────────────────────────────────────────────────────────────

library(quantmod)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

set.seed(42)

# ── 1. DATA LOADING & CLEANING ────────────────────────────────────────────────

TICKERS       <- c("NVDA", "HOOD", "OGN", "VG", "SPY")
LOOKBACK_DAYS <- 365

cat("Pulling data from Yahoo Finance...\n")
getSymbols(TICKERS, src = "yahoo",
           from = Sys.Date() - LOOKBACK_DAYS - 10,
           auto.assign = TRUE, warnings = FALSE)

# Build prices data frame aligned by date
prices <- data.frame(date = index(Cl(get(TICKERS[1]))))
for (t in TICKERS) {
  cl <- Cl(get(t))
  prices[[t]] <- as.numeric(cl[index(cl) %in% prices$date])
}
prices <- na.omit(prices)

# Daily log returns
returns_df <- prices %>%
  mutate(across(all_of(TICKERS), ~ c(NA, diff(log(.x))))) %>%
  na.omit()

cat(sprintf("Observations after cleaning: %d trading days\n\n", nrow(returns_df)))

# ── 2. EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────────

cat("── Summary Statistics (Daily Log Returns) ──────────────────────────────\n\n")
print(summary(returns_df[, TICKERS]))

cat("\n── Annualized Return & Volatility ──────────────────────────────────────\n\n")
eda_stats <- data.frame(
  Ticker     = TICKERS,
  Ann.Return = sapply(TICKERS, function(t) mean(returns_df[[t]]) * 252),
  Ann.Vol    = sapply(TICKERS, function(t) sd(returns_df[[t]])   * sqrt(252))
)
eda_stats$Sharpe <- (eda_stats$Ann.Return - 0.053) / eda_stats$Ann.Vol
print(eda_stats, row.names = FALSE, digits = 4)

# ── EDA Plots ─────────────────────────────────────────────────────────────────

# Histogram of daily returns for each stock
returns_long <- returns_df %>%
  select(all_of(TICKERS)) %>%
  pivot_longer(cols = everything(), names_to = "Ticker", values_to = "Return")

p_hist <- ggplot(returns_long, aes(x = Return, fill = Ticker)) +
  geom_histogram(bins = 50, color = NA, alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ Ticker, nrow = 1, scales = "free_x") +
  scale_fill_manual(values = c("#0F3460","#E94560","#F5A623","#2ECC71","#9B59B6")) +
  labs(title = "Distribution of Daily Log Returns by Stock",
       subtitle = sprintf("%d trading days | Yahoo Finance", nrow(returns_df)),
       x = "Daily Log Return", y = "Frequency") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none", plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))

ggsave("01_return_distributions.png", p_hist, width = 14, height = 4, dpi = 150)
cat("Saved: 01_return_distributions.png\n")

# Cumulative return paths
cum_ret <- returns_df %>%
  mutate(across(all_of(TICKERS), ~ exp(cumsum(.x)))) %>%
  mutate(Date = prices$date[2:nrow(prices)])

cum_long <- cum_ret %>%
  select(Date, all_of(TICKERS)) %>%
  pivot_longer(-Date, names_to = "Ticker", values_to = "CumReturn")

p_cum <- ggplot(cum_long, aes(x = Date, y = CumReturn, color = Ticker)) +
  geom_line(linewidth = 0.9, alpha = 0.9) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  scale_y_continuous(labels = function(x) sprintf("%+.0f%%", (x - 1) * 100)) +
  labs(title = "Cumulative Return Over Observation Period",
       x = "Date", y = "Cumulative Return", color = "Ticker") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("02_cumulative_returns.png", p_cum, width = 9, height = 5, dpi = 150)
cat("Saved: 02_cumulative_returns.png\n\n")

# ── 3. BAYESIAN MODEL ─────────────────────────────────────────────────────────
#
# Model:   r_t ~ Normal(mu, sigma^2),   sigma estimated from data
# Prior:   mu  ~ Normal(mu_0, sigma^2 / kappa_0)
#
# Posterior (Normal-Normal conjugate update):
#   mu_n    = (kappa_0 * mu_0 + n * x_bar) / (kappa_0 + n)
#   kappa_n = kappa_0 + n
#   => posterior: mu | data ~ Normal(mu_n, sigma^2 / kappa_n)

PRIOR_MU    <- 0.0004   # prior mean daily return (~10% annualized)
PRIOR_KAPPA <- 5        # prior strength (equivalent observation count)

cat("── Bayesian Posterior Summary ───────────────────────────────────────────\n\n")

bayes_results <- list()

for (t in TICKERS) {
  r      <- returns_df[[t]]
  n      <- length(r)
  x_bar  <- mean(r)
  sigma  <- sd(r)

  # Posterior parameters
  kappa_n <- PRIOR_KAPPA + n
  mu_n    <- (PRIOR_KAPPA * PRIOR_MU + n * x_bar) / kappa_n
  sigma_n <- sigma / sqrt(kappa_n)

  # 95% credible interval
  ci_lo <- mu_n - 1.96 * sigma_n
  ci_hi <- mu_n + 1.96 * sigma_n

  bayes_results[[t]] <- list(
    ticker  = t,  n = n,
    mu_post = mu_n,  sigma_post = sigma_n,
    sigma   = sigma,
    ci_lo   = ci_lo,  ci_hi = ci_hi,
    ann_ret = mu_n * 252,
    ann_vol = sigma * sqrt(252)
  )
}

# Print posterior summary table
post_df <- do.call(rbind, lapply(bayes_results, function(x) {
  data.frame(
    Ticker      = x$ticker,
    Prior.Mean  = sprintf("%.4f%%", PRIOR_MU * 100),
    Post.Mean   = sprintf("%.4f%%", x$mu_post * 100),
    CI.95.Lo    = sprintf("%.4f%%", x$ci_lo   * 100),
    CI.95.Hi    = sprintf("%.4f%%", x$ci_hi   * 100),
    Ann.Return  = sprintf("%.1f%%",  x$ann_ret * 100),
    Ann.Vol     = sprintf("%.1f%%",  x$ann_vol * 100)
  )
}))
print(post_df, row.names = FALSE)

# ── 4. MONTE CARLO SIMULATION ─────────────────────────────────────────────────

N_SIM    <- 5000
HORIZON  <- 63   # trading days (~3 months)

cat(sprintf("\nRunning %d Monte Carlo paths per stock (%d-day horizon)...\n\n",
            N_SIM, HORIZON))

sim_results <- list()

for (t in TICKERS) {
  b         <- bayes_results[[t]]
  mu_draws  <- rnorm(N_SIM, mean = b$mu_post, sd = b$sigma_post)
  sim_mat   <- matrix(rnorm(N_SIM * HORIZON,
                             mean = rep(mu_draws, HORIZON),
                             sd   = b$sigma),
                      nrow = HORIZON, ncol = N_SIM)
  paths       <- exp(apply(sim_mat, 2, cumsum))
  final       <- paths[HORIZON, ]

  sim_results[[t]] <- list(paths = paths, final = final)

  cat(sprintf("  %-5s  Median: %+.1f%%  |  10th pct: %+.1f%%  |  90th pct: %+.1f%%  |  P(gain): %.0f%%\n",
    t,
    (median(final) - 1) * 100,
    (quantile(final, 0.10) - 1) * 100,
    (quantile(final, 0.90) - 1) * 100,
    mean(final > 1) * 100
  ))
}

# ── 5. RESULTS & PLOTS ────────────────────────────────────────────────────────

# Posterior mean with 95% credible interval
post_plot_df <- data.frame(
  Ticker = names(bayes_results),
  mu     = sapply(bayes_results, `[[`, "mu_post"),
  lo     = sapply(bayes_results, `[[`, "ci_lo"),
  hi     = sapply(bayes_results, `[[`, "ci_hi")
)

p_post <- ggplot(post_plot_df, aes(x = reorder(Ticker, mu), y = mu)) +
  geom_hline(yintercept = 0,        linetype = "dashed", color = "gray60") +
  geom_hline(yintercept = PRIOR_MU, linetype = "dotted", color = "#F5A623",
             linewidth = 0.8) +
  geom_errorbar(aes(ymin = lo, ymax = hi),
                width = 0.25, color = "#0F3460", linewidth = 0.9) +
  geom_point(size = 4, color = "#E94560") +
  annotate("text", x = 0.7, y = PRIOR_MU * 1.6,
           label = "Prior mean", size = 3, color = "#F5A623") +
  scale_y_continuous(labels = percent_format(accuracy = 0.01)) +
  labs(title = "Posterior Mean Daily Return with 95% Credible Interval",
       subtitle = sprintf("Normal-Normal conjugate model | Prior: %.2f%%/day (κ₀ = %d)",
                          PRIOR_MU * 100, PRIOR_KAPPA),
       x = "Ticker", y = "Posterior Mean Daily Return") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("03_posterior_means.png", p_post, width = 7, height = 5, dpi = 150)
cat("\nSaved: 03_posterior_means.png\n")

# Monte Carlo fan chart
fan_data <- do.call(rbind, lapply(TICKERS, function(t) {
  paths <- sim_results[[t]]$paths
  data.frame(
    Ticker = t, Day = seq_len(HORIZON),
    p10 = apply(paths, 1, quantile, 0.10),
    p25 = apply(paths, 1, quantile, 0.25),
    p50 = apply(paths, 1, quantile, 0.50),
    p75 = apply(paths, 1, quantile, 0.75),
    p90 = apply(paths, 1, quantile, 0.90)
  )
}))

p_fan <- ggplot(fan_data, aes(x = Day)) +
  geom_ribbon(aes(ymin = p10, ymax = p90), fill = "#0F3460", alpha = 0.15) +
  geom_ribbon(aes(ymin = p25, ymax = p75), fill = "#0F3460", alpha = 0.25) +
  geom_line(aes(y = p50), color = "#E94560", linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  facet_wrap(~ Ticker, nrow = 1) +
  scale_y_continuous(labels = function(x) sprintf("%+.0f%%", (x - 1) * 100)) +
  labs(title = sprintf("%d-Day Monte Carlo Price Simulation  (%d paths per stock)", HORIZON, N_SIM),
       subtitle = "Red = median path  |  Bands = 25–75th and 10–90th percentiles",
       x = "Trading Days Forward", y = "Simulated Return") +
  theme_minimal(base_size = 11) +
  theme(plot.title  = element_text(face = "bold"),
        strip.text  = element_text(face = "bold", color = "#0F3460"))

ggsave("04_monte_carlo_fan.png", p_fan, width = 14, height = 4, dpi = 150)
cat("Saved: 04_monte_carlo_fan.png\n\n")

# ── 6. KEY RESULTS SUMMARY ────────────────────────────────────────────────────

cat("── Key Results ─────────────────────────────────────────────────────────\n\n")
cat(sprintf("Stocks analyzed:       %s\n", paste(TICKERS, collapse = ", ")))
cat(sprintf("Observation window:    %d trading days\n", nrow(returns_df)))
cat(sprintf("Forecast horizon:      %d trading days (~3 months)\n", HORIZON))
cat(sprintf("Simulation paths:      %d per stock\n", N_SIM))
cat(sprintf("Prior mean (daily):    %.4f%%\n", PRIOR_MU * 100))
cat(sprintf("Prior strength (κ₀):   %d equivalent observations\n\n", PRIOR_KAPPA))
cat("Posterior updates shifted each stock's expected return toward the data.\n")
cat("See plots for credible intervals and simulated price path distributions.\n")
cat("\nDone. All plots saved to working directory.\n")
