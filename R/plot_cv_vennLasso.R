
#' Plot method for cv.vennLasso fitted objects
#'
#' @rdname plot
#' @method plot cv.vennLasso
#' @export
#' @examples
#' set.seed(123)
#' 
#' 
plot.cv.vennLasso <- function(x, sign.lambda = 1, ...) 
{
    # compute total number of selected variables for each
    # tuning parameter 
    nzero <- apply(x$vennLasso.fit$beta[,-1,], 3, function(bb) sum(bb != 0))
    
    xlab <- expression(log(lambda))
    if(sign.lambda<0)xlab <- paste("-", xlab, sep="")
    plot.args = list(x    = sign.lambda * log(x$lambda),
                     y    = x$cvm,
                     ylim = range(x$cvup, x$cvlo),
                     xlab = xlab,
                     ylab = x$name,
                     type = "n")
    new.args <- list(...)
    if(length(new.args)) plot.args[names(new.args)] <- new.args
    do.call("plot", plot.args)
    error.bars(sign.lambda * log(x$lambda), 
               x$cvup, 
               x$cvlo, width = 0.005)
    points(sign.lambda*log(x$lambda), x$cvm, pch=20, col="dodgerblue")
    axis(side   = 3,
         at     = sign.lambda * log(x$lambda),
         labels = paste(nzero), tick=FALSE, line=0, ...)
    abline(v = sign.lambda * log(x$lambda.min), lty=2, lwd = 2, col = "firebrick3")
    abline(v = sign.lambda * log(x$lambda.1se), lty=2, lwd = 2, col = "firebrick1")
    title(x$name, line = 2.5, ...)
    
}
