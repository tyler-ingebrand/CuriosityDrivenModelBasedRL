using Flux
using Flux.Optimise





f(x) = sum(x)^2
x = [10.0]
ps = Flux.params(x)
opt = Descent(0.01)

 for i in 1:1000

    gs = gradient(() -> f(x), ps)
    Flux.Optimise.update!(opt, ps, gs)
    println(x)
end
