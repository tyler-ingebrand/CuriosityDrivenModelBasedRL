mutable struct KnownDynamicsAgent <: Agent
    # known in this case
    dynamics # function mapping s,a -> s
    
    # known functions 
    reward # function mapping s,a,s -> R
    value # function mapping s -> R
    
    # Rules for optimizing
    optimizer # gradient based optimizer
    rollout_steps # number of steps to look into the future
    action_space
end

(::KnownDynamicsAgent)(state) = choose_action(agent, state)
# learn not defined, no update needed since its known


# helpers
function predict_value(agent::KnownDynamicsAgent, initial_state, actions)
    sum_value = 0
    state = initial_state

    # for each action in our lookahead, account for reward recieved
    for action in actions
        next_state = agent.dynamics(state, action)
        sum_value += agent.reward(state, action, next_state)
        state = next_state
    end

    # account for final state's infintie horizon value
    sum_value += agent.value(state)

    return sum_value
end

function choose_action(agent::KnownDynamicsAgent, initial_state)
    # init actions (0s or guess)
    num_actions = typeof(agent.action_space) == Space ? length(agent.action_space) : 1
    actions = zeros(agent.rollout_steps, num_actions)

    # optimize based on predicted value, optimzer, and initial state 
    ps = Flux.params(actions)

    for i in 1:1000

        gs = gradient(() -> -predict_value(agent, initial_state, actions), ps)
        Flux.Optimise.update!(agent.optimizer, ps, gs)

        # project value back into action space
        actions = clamp.(actions, agent.action_space.left, agent.action_space.right)
    end

    return actions[1, :]
end