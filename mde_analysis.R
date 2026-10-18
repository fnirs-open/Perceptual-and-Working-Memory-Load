library(readr)
library(lme4)
library(simr)
library(purrr)
library(dplyr)
library(ggplot2)

# Inputs exported from 4_final_analysis_aligned.ipynb:
# channel-level data (theta x 1e6, centered +-0.5 coding) and the representative HbO channels
# (closest to the 10th percentile, median and 90th percentile of residual SD)
df       <- read_csv("data/simr_input.csv")
channels <- read_csv("data/simr_channels.csv") |> filter(Chroma == "hbo")

# Settings
effect_sizes <- seq(0.5, 4, by = 0.25)  # fixed-effect coefficient, theta x 1e6; 0 checks type I error
nsim         <- 500
m_tests      <- 132                   # BH family for HbO in the notebook (44 channels x 3 terms)
alphas       <- c(uncorrected = 0.05, corrected = 0.05 / m_tests)
set.seed(1)

terms <- c(
  "Load" = "Load_c",
  "Memory" = "Memory_c",
  "Interaction" = "Load_c:Memory_c"
)

# Simulate once per channel, term and effect size, keeping every p-value,
# so power at both alphas comes from the same simulations
pval_df <- pmap_dfr(channels, function(Chroma, Variability, ch_name, ...) {

  d <- df[df$ch_name == ch_name, ]

  # Same model as the notebook: main effects + interaction, random intercept per subject
  m <- lmer(
    theta ~ Load_c * Memory_c + (1 | subject),
    data = d
  )

  message(sprintf("%s (%s, %s): sigma_e = %.3f", ch_name, Chroma, Variability, sigma(m)))

  imap_dfr(terms, function(term, label) {

    map_dfr(effect_sizes, function(eff) {

      # Replace the coefficient for this term; variance components stay as fitted
      m_tmp <- m
      fixef(m_tmp)[term] <- eff

      # Kenward-Roger test (requires pbkrtest), appropriate for small numbers of subjects
      ps <- powerSim(
        m_tmp,
        test = fixed(term, method = "kr"),
        nsim = nsim,
        progress = FALSE
      )

      data.frame(
        Chroma = Chroma,
        Variability = Variability,
        ch_name = ch_name,
        sigma_e = sigma(m),
        term = label,
        effect_size = eff,
        pval = ps$pval
      )
    })
  })
})

write_csv(pval_df, "data/simr_pvals.csv")

# Power and 95% Clopper-Pearson CI at each alpha; failed fits (NA p-values) are excluded
power_df <- imap_dfr(alphas, function(a, lab) {
  pval_df |>
    filter(!is.na(pval)) |>
    group_by(Chroma, Variability, ch_name, sigma_e, term, effect_size) |>
    summarise(
      n_sims = n(),
      successes = sum(pval < a),
      .groups = "drop"
    ) |>
    mutate(
      alpha_label = lab,
      alpha = a,
      power = successes / n_sims,
      lower = map2_dbl(successes, n_sims, ~ binom.test(.x, .y)$conf.int[1]),
      upper = map2_dbl(successes, n_sims, ~ binom.test(.x, .y)$conf.int[2])
    )
})

power_df <- power_df |>
  mutate(
    term = factor(term, levels = names(terms)),
    Variability = factor(Variability, levels = c("p10", "median", "p90")),
    alpha_label = factor(
      alpha_label,
      levels = names(alphas),
      labels = c(
        "α = 0.05 (uncorrected)",
        sprintf("α = 0.05 / %d (corrected)", m_tests)
      )
    )
  )

write_csv(power_df, "data/simr_power.csv")

if (any(power_df$n_sims < nsim)) {
  message("Some simulations failed; see n_sims in data/simr_power.csv")
}

# First grid point where each curve reaches 80% power, marked on the plots
power80_df <- power_df |>
  filter(power >= 0.80) |>
  group_by(Chroma, Variability, ch_name, term, alpha_label) |>
  slice_min(effect_size, n = 1) |>
  ungroup()

write_csv(power80_df, "data/simr_power80.csv")
print(power80_df |> select(Chroma, Variability, ch_name, term, alpha_label, effect_size, power), n = Inf)

# One figure per alpha: columns = term, colour = channel variability
for (alpha_key in names(alphas)) {

  pd_sub      <- filter(power_df, alpha == alphas[[alpha_key]])
  power80_sub <- filter(power80_df, alpha == alphas[[alpha_key]])

  ch_labels <- pd_sub |>
    distinct(Variability, ch_name) |>
    arrange(Variability) |>
    mutate(label = sprintf("%s: %s", Variability, ch_name))
  ch_labels <- setNames(ch_labels$label, as.character(ch_labels$Variability))

  p <- ggplot(
    pd_sub,
    aes(effect_size, power, colour = Variability, fill = Variability)
  ) +
    geom_hline(yintercept = 0.80, linetype = "dashed", colour = "grey50") +
    geom_hline(yintercept = alphas[[alpha_key]], linetype = "dotted", colour = "grey70") +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, colour = NA) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 1.5) +
    geom_point(data = power80_sub, shape = 21, size = 3.5, stroke = 1.2, fill = "white") +
    geom_text(
      data = power80_sub,
      aes(label = sprintf("%.2f", effect_size)),
      hjust = -0.25, vjust = 1.6, size = 4, fontface = "bold", show.legend = FALSE
    ) +
    facet_wrap(~ term, nrow = 1) +
    scale_colour_manual(values = c(p10 = "#2C6E9B", median = "#3A9A5B", p90 = "#C8553D"),
                        labels = ch_labels) +
    scale_fill_manual(values = c(p10 = "#2C6E9B", median = "#3A9A5B", p90 = "#C8553D"),
                      labels = ch_labels) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), labels = scales::percent) +
    coord_cartesian(ylim = c(0, 1)) +
    theme_bw(base_size = 14) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey95", colour = NA),
      strip.text = element_text(face = "bold")
    ) +
    labs(
      title = sprintf("Simulated power - HbO, %s", as.character(pd_sub$alpha_label[1])),
      x = "True fixed-effect coefficient (theta × 1e6)",
      y = "Estimated power",
      colour = "Residual SD percentile",
      fill = "Residual SD percentile"
    )

  print(p)
  ggsave(sprintf("img/power_curve_hbo_%s.png", alpha_key), p, width = 13, height = 5.5, dpi = 200)
}
