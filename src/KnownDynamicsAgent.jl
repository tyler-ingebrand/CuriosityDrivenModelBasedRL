

mutable struct KnownDynamicsAgent
    dynamics # function mapping s,a -> s
    reward # function mapping s,a,s -> R
    value # function mapping s -> R
    optimizer # gradient based optimizer
end



