library(ggplot2)

df <- data.frame(
  company = c("Apple", "Microsoft", "Amazon", "Google", "Facebook"),
  revenue = c(74.6, 58.4, 386.1, 181.2, 70.7)  # 模拟的营收数据（以亿美元为单位）
)

ggplot(data = df) +
  geom_point(aes(y = company, x = revenue),
             size = 2.5, fill = "#eb3b6c", shape = 21) +
  scale_x_continuous(limits = c(0, 400), expand = c(0, 0)) +  # 修改x轴范围
  coord_cartesian(clip = "off") +
  theme_minimal() +
  theme(
    axis.title = element_blank(),
    panel.grid = element_blank(),
    panel.grid.major.x = element_line(color = "grey", linetype = "dashed"),
    plot.background = element_rect(fill = "white", color = "white")
  )
