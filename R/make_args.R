
make_args <- function( biter=1000, burnin=500, badapt1=100, badapt2=100, nchain=1,
    max_depth=10, max_depth_adapt=10, delta=0.80, gamma=0.05, t0=10, kappa=0.75 )
{
    
    #- make adapt - list:
    adapt <- list( delta=delta, gamma=gamma, t0=t0, kappa=kappa)
    #- ...
    return( list( biter=biter, 
        burnin=burnin, 
        badapt1=badapt1,
        badapt2=badapt2,
        nchain=nchain, 
        epsilon=0.10, L=20,
        max_depth=max_depth, max_depth_adapt=max_depth_adapt, adapt=adapt ) )
}