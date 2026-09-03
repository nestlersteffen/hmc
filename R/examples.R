
# a simple regression example

make_regression <- function( y=NULL, X=NULL, lambda2=NULL, a=NULL, b=NULL )
{
    function( theta ) {
        #- get some information:
        p <- length( theta )
        n <- dim(X)[1]
        #- make parms:
        beta   <- theta[1:(p-1)]
        phi    <- theta[p]
        sigma2 <- exp( 2*phi )
        #- compute nll:
        ey <- y - X%*%beta
        #- compute posterior components:
        sum_ey_sq   <- sum(ey*ey)
        sum_beta_sq <- sum(beta*beta)
        ll   <- -n * phi - sum_ey_sq/( 2*sigma2 )
        lp_b <- -sum_beta_sq/( 2*lambda2 )
        lp_s <- -2*(a+1)*phi - b/sigma2
        jac  <- 2*phi
        #- compute gradient:
        g_beta <- (1/sigma2) * ( t( X ) %*% ey) - (beta/lambda2)
        g_phi  <- -n + sum_ey_sq/sigma2 - 2*(a + 1) + (2*b/sigma2) + 2
        #- make output:
        res <- list( fn=-1*(ll + lp_b + lp_s + jac), gr=-1*c(g_beta,g_phi) )
        return( res )
    }
}
