using ReinforcementLearning
using Flux
using Flux.Optimise
using Plots



include("src/Interface.jl")
include("src/KnownDynamicsAgent.jl")
include("pendelum_functions.jl")

# create env
env = PendulumEnv()
reset!(env)

# create agent
agent = KnownDynamicsAgent( pendelum_dynamics,   
                            pendelum_reward ,
                            pendelum_value ,
                            Descent(01.0),
                            30, # lookahead steps 
                            action_space(env)
                        )

# simulate episode
total_r = 0.0
while !is_terminated(env)
    s = state(env)
    a = agent(s)
    env(a[1])
    global total_r += reward(env)
    plot(env)
    println("S = ", s, ", A = ", a)
    #println(atan(s[2]/s[1]) + pi)
end
println("Total R = ", total_r)