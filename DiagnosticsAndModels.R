install.packages("glmnet", repos='http://cran.us.r-project.org')
install.packages("ISLR", repos='http://cran.us.r-project.org')
install.packages("ISLR2")
install.packages("lmtest")
library(ISLR2)
library(glmnet, quietly = TRUE)
library(ISLR, quietly = TRUE)
library(lmtest)
FinalDF <- read.csv("model_filled.csv", header = TRUE)
TestOLS <- lm(y ~ ., data = FinalDF)
x <- model.matrix(y ~., data = FinalDF)[, -1]
y <- FinalDF$y
grid.lambda <- 10^seq(10, -2, length = 100)
ridge.model <- glmnet(x, y, alpha = 0, lambda = grid.lambda)
plot(ridge.model)
ridge.model$lambda[50]
coef(ridge.model)[, 50]
sqrt(sum(coef(ridge.model)[-1, 50]^2))
ell2.norm <- numeric()
for(i in 1:length(grid.lambda)){
  ell2.norm[i] <- sqrt(sum(coef(ridge.model)[-1, i]^2))
}
plot(x = grid.lambda, y = ell2.norm, xlab = expression(lambda), ylab = "L2 Norm", xlim = c(0,50))
set.seed(1)
train <- sample(1:nrow(x), nrow(x) / 2)
test <- (-train)
y.train <- y[train]
y.test <- y[test]
ridge.model.train <- glmnet(x[train, ], y.train, alpha = 0, lambda = grid.lambda)
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y.train, alpha = 0)
plot(cv.out)
best.lambda <- cv.out$lambda.min
best.lambda
plot(cv.out)
abline(v = log(best.lambda), col = "blue", lwd = 2)
ridge.pred <- predict(ridge.model.train, s = best.lambda, newx = x[test, ])
mspe.ridge <- mean((ridge.pred - y.test)^2)
mspe.ridge
final.model <- glmnet(x, y, alpha = 0, lambda = best.lambda)
Coef.Ridge <- coef(final.model)[1:92, ]
Coef.Ridge
lasso.model <- glmnet(x, y, alpha = 1, lambda = grid.lambda)
lasso.model.train <- glmnet(x[train, ], y.train, alpha = 1, lambda = grid.lambda)
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y.train, alpha = 1)
best.lambda <- cv.out$lambda.min
lasso.pred <- predict(lasso.model.train, s = best.lambda, newx = x[test,])
mspe.lasso <- mean((lasso.pred - y.test)^2)
mspe.lasso
final.model <- glmnet(x, y, alpha = 1, lambda = best.lambda)
Coef.Lasso <- coef(final.model)[1:92,]
Coef.Lasso
which(Coef.Lasso > 0) # 4 coefs are not 0: Intercept, finance, hours_out, education_ranking
shapiro_result <- shapiro.test(TestOLS$residuals)
shapiro_result #p-value=0.004 so normality does not hold
bp_results <- bptest(TestOLS)
bp_results #p-value=0.2564 so constant variance is possible 
hats <- as.data.frame(hatvalues(TestOLS))
indx.high.leverage <- which(hatvalues(TestOLS) > 2*92/103)
indx.high.leverage # 0 leverage points
infl_tab <- influence.measures(TestOLS)
global_influence_measures <- infl_tab$infmat
global_influence_measures <- global_influence_measures[, 93:96]
global_is_influential <- infl_tab$is.inf
global_is_influential <- global_is_influential[, 93:96]
sum(global_is_influential[, 4]) # 0 influential points
which(rowSums(global_is_influential[, 1:3]) >= 2) # 0 points with 2 or more influential indicators 
