
set.seed(1)
data <- rnorm(500, mean=0, sd=1)

par(mfrow=c(1,1))
qqnorm(data, main="QQ Plot")
qqline(data, col="blue")

