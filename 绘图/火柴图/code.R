library(tidyverse)

set.seed(123)
companies <- c("Company A", "Company B", "Company C", "Company D", "Company E")
revenues <- runif(length(companies), min = 10, max = 70)

df <- tibble(company = factor(companies, levels = companies),
             revenue = revenues)

df <- df %>% arrange(revenue)

ggplot(data = df) +
  geom_segment(aes(y = company, x = 0, xend = revenue),
               size = 1, color = "grey30") +
  geom_point(aes(y = company, x = revenue),
             size = 3.5, fill = "#eb3b6c", shape = 21) +
  scale_x_continuous(limits = c(0, 75), expand = c(0, 0)) +
  theme_minimal() +
  theme(axis.title = element_blank(),
        panel.grid = element_blank(),
        plot.background = element_rect(fill = "white", color = "white"))
