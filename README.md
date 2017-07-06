



# vennLasso
Hierarchical variable selection for models stratified on binary factors 


## Installation and Help Files


Install using the **devtools** package:


```r
devtools::install_github("jaredhuling/vennLasso")
```


or by cloning and building.

Load the **vennLasso** package:

```r
library(vennLasso)
```

Access help file for the main fitting function ``vennLasso()`` by running:


```r
?vennLasso
```

Help file for cross validation function ``cv.vennLasso()`` can be accessed by running:


```r
?cv.vennLasso
```

## A Quick Example

Simulate heterogeneous data:


```r
dat.sim <- genHierSparseData(ncats = 3,  # number of stratifying factors
                             nvars = 25, # number of variables
                             nobs = 150, # number of observations per strata
                             nobs.test = 10000,
                             hier.sparsity.param = 0.5,
                             prop.zero.vars = 0.75, # proportion of variables
                                                   # zero for all strata
                             snr = 0.5,  # signal-to-noise ratio
                             family = "gaussian")

# design matrices
x        <- dat.sim$x
x.test   <- dat.sim$x.test

# response vectors
y        <- dat.sim$y
y.test   <- dat.sim$y.test

# binary stratifying factors
grp      <- dat.sim$group.ind
grp.test <- dat.sim$group.ind.test
```

Fit vennLasso model with tuning parameter selected with 5-fold cross validation:


```r
fit.adapt <- cv.vennLasso(x, y,
                          grp,
                          adaptive.lasso = TRUE,
                          nlambda        = 50,
                          family         = "gaussian",
                          standardize    = FALSE,
                          intercept      = TRUE,
                          nfolds         = 5)
```

Predict response for test data:


```r
preds.vl <- predict(fit.adapt, x.test, grp.test, s = "lambda.min",
                    type = 'response')
```

Evaluate mean squared error:


```r
mean((y.test - preds.vl) ^ 2)
```

```
## [1] 0.5733229
```


```r
mean((y.test - mean(y.test)) ^ 2)
```

```
## [1] 0.8359014
```


Compare with naive model with all interactions between covariates and stratifying binary factors:

```r
df.x <- data.frame(y = y, x = x, grp = grp)
df.x.test <- data.frame(x = x.test, grp = grp.test)

# create formula for interactions between factors and covariates
form <- paste("y ~ (", paste(paste0("x.", 1:ncol(x)), collapse = "+"), ")*(grp.1*grp.2*grp.3)" )
```

Fit linear model and generate predictions for test set:

```r
lmf <- lm(as.formula(form), data = df.x)

preds.lm <- predict(lmf, df.x.test)
```

Evaluate mean squared error:


```r
mean((y.test - preds.lm) ^ 2)
```

```
## [1] 0.6810668
```

```r
mean((y.test - preds.vl) ^ 2)
```

```
## [1] 0.5733229
```




