[![version](http://www.r-pkg.org/badges/version/vennLasso)](https://cran.r-project.org/package=vennLasso)

# vennLasso

The `vennLasso` package provides methods for hierarchical variable selection for models with covariate effects stratified by multiple binary factors.




## Installation and Help Files

The `vennLasso` package can be installed from CRAN using:

``` r
install.packages("vennLasso")
```

The development version can be installed using the **devtools** package:


``` r
devtools::install_github("jaredhuling/vennLasso")
```


or by cloning and building.

Load the **vennLasso** package:

``` r
library(vennLasso)
```

Access help file for the main fitting function ``vennLasso()`` by running:


``` r
?vennLasso
```

Help file for cross validation function ``cv.vennLasso()`` can be accessed by running:


``` r
?cv.vennLasso
```

## A Quick Example

Simulate heterogeneous data:


``` r
set.seed(100)
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

Inspect the populations for each strata:


``` r
plotVenn(grp)
```

![](man/figures/README-unnamed-chunk-7-1.png)<!-- -->

Fit vennLasso model with tuning parameter selected with 5-fold cross validation:


``` r
fit.adapt <- cv.vennLasso(x, y,
                          grp,
                          adaptive.lasso = TRUE,
                          nlambda        = 50,
                          family         = "gaussian",
                          standardize    = FALSE,
                          intercept      = TRUE,
                          nfolds         = 5)
```


Plot selected variables for each strata (not run): 


``` r
library(igraph)
```

```
## 
## Attaching package: 'igraph'
```

```
## The following objects are masked from 'package:stats':
## 
##     decompose, spectrum
```

```
## The following object is masked from 'package:base':
## 
##     union
```

``` r
plotSelections(fit.adapt)
```

```{=html}
<div class="visNetwork html-widget html-fill-item" id="htmlwidget-05fb5a2d29a2ee413814" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-05fb5a2d29a2ee413814">{"x":{"nodes":{"id":[1,2,3,5,4,6,7,8],"label":["0,0,0","0,0,1","0,1,0","1,0,0","0,1,1","1,0,1","1,1,0","1,1,1"],"value":[10,4,5,2,8,7,7,12],"title":["<p>Num vars selected: 10<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: 0.11208<br> <b>V1<\/b>: -0.11964<br> <b>V2<\/b>: -0.42585<br> <b>V4<\/b>: -0.05327<br> <b>V5<\/b>: -0.03519<br> <b>V7<\/b>: 0.62001<br> <b>V14<\/b>: 0.08411<br> <b>V18<\/b>: 0.05262<br> <b>V21<\/b>: -0.04707<br> <b>V24<\/b>: -0.08219<br><\/div><\/div>","<p>Num vars selected: 4<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: -0.0812<br> <b>V1<\/b>: -0.38274<br> <b>V6<\/b>: -0.0331<br> <b>V20<\/b>: -0.00579<br><\/div><\/div>","<p>Num vars selected: 5<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: -0.03698<br> <b>V1<\/b>: -0.189<br> <b>V3<\/b>: 0.26919<br> <b>V9<\/b>: 0.00084<br> <b>V20<\/b>: 0.03832<br><\/div><\/div>","<p>Num vars selected: 2<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: -0.0156<br> <b>V20<\/b>: 0.00112<br><\/div><\/div>","<p>Num vars selected: 8<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: 0.03329<br> <b>V1<\/b>: 0.23304<br> <b>V3<\/b>: 0.33573<br> <b>V5<\/b>: 0.28517<br> <b>V6<\/b>: 0.30775<br> <b>V9<\/b>: -0.00108<br> <b>V17<\/b>: 0.00244<br> <b>V20<\/b>: 0.09257<br><\/div><\/div>","<p>Num vars selected: 7<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: 0.04388<br> <b>V1<\/b>: 0.24848<br> <b>V2<\/b>: -0.16717<br> <b>V6<\/b>: -0.45723<br> <b>V13<\/b>: 0.02899<br> <b>V17<\/b>: -5e-05<br> <b>V20<\/b>: 0.01159<br><\/div><\/div>","<p>Num vars selected: 7<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: 0.01603<br> <b>V1<\/b>: -0.4504<br> <b>V2<\/b>: 0.36559<br> <b>V3<\/b>: -0.45971<br> <b>V9<\/b>: -2e-04<br> <b>V17<\/b>: -0.01432<br> <b>V20<\/b>: -0.01554<br><\/div><\/div>","<p>Num vars selected: 12<\/p><div class='rPartvisNetworkTooltipShowhim' style='color:blue;'> \n                                <U>Coefficients<\/U><div class='rPartvisNetworkTooltipShowme' \n                                style='color:black;overflow-y: scroll;height: 150px;'><b>(Intercept)<\/b>: -0.00861<br> <b>V1<\/b>: -0.48637<br> <b>V2<\/b>: -0.33566<br> <b>V3<\/b>: -0.23978<br> <b>V5<\/b>: 0.25092<br> <b>V6<\/b>: -0.46224<br> <b>V7<\/b>: 0.01291<br> <b>V9<\/b>: 0.00482<br> <b>V13<\/b>: 0.02065<br> <b>V17<\/b>: -0.03555<br> <b>V20<\/b>: 0.04579<br> <b>V23<\/b>: 0.00046<br><\/div><\/div>"],"x":[-0.4444444444444443,-1,-0.3333333333333334,0.3333333333333333,-0.6666666666666667,0.3333333333333333,1,0.2222222222222223],"y":[-1,1,1,1,0,0,0,-1]},"edges":{"from":[4,6,4,7,8,6,7,8,8],"to":[2,2,3,3,4,5,5,6,7],"value":[4,4,5,5,8,2,2,7,7]},"nodesToDataframe":true,"edgesToDataframe":true,"options":{"width":"100%","height":"100%","nodes":{"shape":"dot","mass":1,"physics":false,"font":{"size":25}},"manipulation":{"enabled":false},"edges":{"smooth":true},"physics":{"stabilization":false},"interaction":{"hover":true,"tooltipDelay":0,"zoomSpeed":1}},"groups":null,"width":null,"height":null,"idselection":{"enabled":true,"style":"width: 150px; height: 26px","useLabels":true,"main":"Select by id"},"byselection":{"enabled":false,"style":"width: 150px; height: 26px","multiple":false,"hideColor":"rgba(200,200,200,0.5)","highlight":false},"main":null,"submain":null,"footer":null,"background":"rgba(0, 0, 0, 0)","igraphlayout":{"type":"full"},"highlight":{"enabled":true,"hoverNearest":true,"degree":1,"algorithm":"all","hideColor":"rgba(200,200,200,0.5)","labelOnly":true},"collapse":{"enabled":false,"fit":false,"resetHighlight":true,"clusterOptions":null,"keepCoord":true,"labelSuffix":"(cluster)"},"tooltipStay":300,"tooltipStyle":"position: fixed;visibility:hidden;padding: 5px;white-space: nowrap;font-family: verdana;font-size:14px;font-color:#000000;background-color: #f5f4ed;-moz-border-radius: 3px;-webkit-border-radius: 3px;border-radius: 3px;border: 1px solid #808074;box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);"},"evals":[],"jsHooks":[]}</script>
```

Predict response for test data:


``` r
preds.vl <- predict(fit.adapt, x.test, grp.test, s = "lambda.min",
                    type = 'response')
```

Evaluate mean squared error:


``` r
mean((y.test - preds.vl) ^ 2)
```

```
## [1] 0.6837243
```


``` r
mean((y.test - mean(y.test)) ^ 2)
```

```
## [1] 1.011026
```


Compare with naive model with all interactions between covariates and stratifying binary factors:

``` r
df.x <- data.frame(y = y, x = x, grp = grp)
df.x.test <- data.frame(x = x.test, grp = grp.test)

# create formula for interactions between factors and covariates
form <- paste("y ~ (", paste(paste0("x.", 1:ncol(x)), collapse = "+"), ")*(grp.1*grp.2*grp.3)" )
```

Fit linear model and generate predictions for test set:

``` r
lmf <- lm(as.formula(form), data = df.x)

preds.lm <- predict(lmf, df.x.test)
```

Evaluate mean squared error:


``` r
mean((y.test - preds.lm) ^ 2)
```

```
## [1] 0.8056107
```

``` r
mean((y.test - preds.vl) ^ 2)
```

```
## [1] 0.6837243
```




