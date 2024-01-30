using Random;

using DifferentialEquations;
using SciMLSensitivity;

using ComponentArrays
using Suppressor;
using Lux;
using NNlib: sigmoid;
using Flux.Losses: mae;
using Statistics;
using JLSO;

println("----Loading Experiment Config----");

"""
    Hybrid Model of a non-isolated cell signalinga pathway.
"""
function ude_dynamics!(du, u, θ, nn_st, t)
    
    missingInputs = neuralNetwork(u, θ, nn_st)[1];
    x1, x2, x1x2, x3, x4, x3x4, x5, x3x5, x6, x2x6, _ = @view u[:];

    kf1 = 0.015f0;
    kr1 = 0.100f0;
    kcat1 = 0.003f0;
    kf2 = 0.099f0;
    kr2 = 0.115f0;
    kcat2 = 0.085f0;
    kf3 = 0.089f0;
    kr3 = 0.05f0;
    kcat3 = 0.15f0;
    kf4 = 0.25f0;
    kr4 = 0.4325f0;
    kcat4 = 0.0150f0;
    
    #  x1
    du[1]  = -1 * (kf1 * x1 * x2)  + (kr1 * x1x2 + kcat1 * x1x2) + missingInputs[1];
    #  x2
    du[2]  = -1 * (kf1 * x1 * x2 + kf4 * x2 * x6) + (kr1 * x1x2 + kr4 * x2x6) + missingInputs[2];
    #  x1x2
    du[3]  = -1 * (kcat1 * x1x2 + kr1 * x1x2)  + (kf1 * x1 * x2);
    #  x3
    du[4]  = -1 * (kf2 * x3 * x4 + kf3 * x3 * x5) + (kcat1 * x1x2 + kcat2 * x3x4 + kcat3 * x3x5 + kr2 * x3x4 + kr3 * x3x5) + missingInputs[3];
    #  x4
    du[5]  = -1 * (kf2 * x3 * x4) + (kr2 * x3x4) + missingInputs[4];
    #  x3x4
    du[6]  = -1 * (kcat2 * x3x4 + kr2 * x3x4) + (kf2 * x3 * x4);
    #  x5
    du[7]  = -1 * (kf3 * x3 * x5) + (kcat2 * x3x4 + kr3 * x3x5) + missingInputs[5];
    #  x3x5
    du[8]  = -1 * (kcat3 * x3x5 + kr3 * x3x5) + (kf3 * x3 * x5);
    #  x6
    du[9]  = -1 * (kf4 * x2 * x6) + (kcat3 * x3x5 + kcat4 * x2x6 + kr4 * x2x6) + missingInputs[6];
    #  x2x6
    du[10] = -1 * (kcat4 * x2x6 + kr4 * x2x6) + (kf4 * x2 * x6);
    #  x7
    du[11] = (kcat4 * x2x6) + missingInputs[7];
end;

"""
    Function to simulate the model with a initial condition 
    from the train set
"""
function prob_func(prob, i, repeat)
    remake(prob, u0 = conditions[i][:u0])
end;

"""
    Function to simulate the model with a initial condition 
    from the validation set
"""
function valprob_func(prob, i, repeat)
    remake(prob, u0 = conditions[train_size + i][:u0])
end;

"""
    Function to simulate the model with a initial condition 
    from the test set
"""
function testprob_func(prob, i, repeat)
    remake(prob, u0 = conditions[train_size + val_size + i][:u0])
end;


rng = Random.default_rng();

neuralNetwork = Chain(
    Dense(11 => 7, sigmoid),
    Dense(7 => 7)
);

nn_p, nn_st = Lux.setup(rng, neuralNetwork);
nn_params = ComponentVector{Float64}(nn_p);
nn_p =  nn_params .* 0 + Float64(1e-4) * randn(rng, eltype(nn_params), size(nn_params));

u0      = zeros(Float32, 11);
tspan   = (0.0f0, 100.0f0);

odeFunction!(du, u, p, t) = ude_dynamics!(du, u, p, nn_st, t);
model       = ODEProblem{true, SciMLBase.FullSpecialize}(odeFunction!, u0, tspan, nn_p);

conditions  = Dict[]; # set of initial conditions
n_fails     = 0;

colors = [:red, :blue, :brown, :magenta, :green, :orange, :navyblue, :black, :pink, :plum4, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray,:gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray];
species_name = [
    "x1",     "x2",  "x1x2",   "x3",    "x4",  "x3x4",   "x5", "x3x5",     "x6", "x2x6", "x7"
];

# @var experimentConfig is a global variable 
# that has the info about the experiment
savaAt = experimentConfig["savePointsAt"];

# configure wich chemical species will be observed by the model
observedSpecies = (
    haskey(experimentConfig, "observedSpecies") && experimentConfig["observedSpecies"] isa Vector
) ? experimentConfig["observedSpecies"] : 1:11; 

# Create the ensemble problem to simulate the model in parallel
ensembleProblem = EnsembleProblem(model, prob_func = prob_func);            # train set 
valEnsembleProblem = EnsembleProblem(model, prob_func = valprob_func);      # validation set 
testEnsembleProblem = EnsembleProblem(model, prob_func = testprob_func);    # test set

"""
    Function to set the initial conditions of 
    the experiment.
"""
set_conditions(values) = begin
    global conditions = values;
end

# at wich iteration after the validation loss increased 
# it should be checked and stopped if it did not decreased
check_val_iter      = 100;

min_val_loss_i      = 1;
min_val_loss_v      = Inf;
last_index_callback = 0;

# extra step to run the optimization
otp_extra_step      = 0;

"""
    Callback of the optimizer
"""
function callback(p, l)

    global last_index_callback += 1;

    valLoss   = Float32.(validation_loss(p));
    testLoss  = Float32.(testset_loss(p));
    trainLoss = Float32.(l);

    open("$(results_folder)/valLoss.csv", "a+")   do io writedlm(io, valLoss, ',');   end;
    open("$(results_folder)/testLoss.csv", "a+")  do io writedlm(io, testLoss, ',');  end;
    open("$(results_folder)/trainLoss.csv", "a+") do io writedlm(io, trainLoss, ','); end;

    # check if the validation loss continues to increase after (check_val_iter + otp_extra_step) steps
    is_to_check_loss_val = (last_index_callback - min_val_loss_i) > check_val_iter + otp_extra_step;

    # Saves the best parameter till validation loss increased.
    if (min_val_loss_v > valLoss)

        global min_val_loss_i = last_index_callback;
        global min_val_loss_v = valLoss;
        global otp_extra_step = 0;

        JLSO.save(
            "$(val_param_folder)/val_param.jlso", 
            :ude_parameters => p
        );
    end;

    print("[$(last_index_callback)] MinValLoss[$(min_val_loss_i)]: $(min_val_loss_v) -- Train: $(trainLoss), Validation: $(valLoss), Test: $(testLoss)\r");
    
    return is_to_check_loss_val && min_val_loss_v > trainLoss;
end;


function predict(θ, initial_condition)
    tmp_prob = remake(model, u0=initial_condition, p=θ);

    tmp_sol = solve(
        tmp_prob, 
        AutoVern7(Rodas4()), 
        abstol=1e-6, 
        reltol=1e-6,
        saveat = savaAt,
        save_idxs = observedSpecies,
        sensealg = SciMLSensitivity.QuadratureAdjoint(
            autojacvec=SciMLSensitivity.ReverseDiffVJP()
        )
    );

    return tmp_sol
end

function test_loss(p, condition)
    sol = predict(p, condition[:u0]);

    if SciMLBase.successful_retcode(sol.retcode)
        X̂ = Array(sol);
        return mae(X̂, condition[:X]);
    else
        return Inf;
    end
end;

function loss(p) # the training loss
    @suppress_err begin
        return evaluate_loss(p, ensembleProblem, train_size, 0, train_size);
    end;
end;

function validation_loss(p)
    @suppress_err begin
        return evaluate_loss(p, valEnsembleProblem, val_size, train_size, val_size);
    end;
end;

function testset_loss(p)
    @suppress_err begin
        return evaluate_loss(p, testEnsembleProblem, test_size, train_size + val_size, test_size);
    end;
end;

function evaluate_loss(p, ensembleProblem, trajectories, step, N)
    try
        sim = solve(
            ensembleProblem, 
            AutoVern7(Rodas4()), 
            saveat = savaAt,
            save_idxs = observedSpecies,
            abstol=1e-6, 
            reltol=1e-6, 
            sensealg = SciMLSensitivity.QuadratureAdjoint(
                autojacvec=SciMLSensitivity.ReverseDiffVJP()
            ),    
            trajectories = trajectories,
            p = p
        );

        error = convert(eltype(p), 1e-3) * sum(abs, p) ./length(p);

        for i in 1:N
            condition = conditions[step + i];
            sol = sim[i];

            if SciMLBase.successful_retcode(sol.retcode)
                X̂ = Array(sol);
                error += mae(X̂, condition[:X]);
            else
                return Inf;
            end
        end;
    
        return error;
    catch e
        error_msg = sprint(showerror, e)
        println("Error => LossFunction: $(error_msg)")
        return Inf;
    end;
end;

"""
    This function configures the initial weights of
    the NN. It initializes the weights that do not leads 
    to numerical instability at the first simulation
"""
function config_initial_weights()

    error = loss(nn_p);

    println("----First test...");

    while error == Inf
        println("----Try again----");
        global nn_p, nn_st = Lux.setup(rng, neuralNetwork);
        nn_params = ComponentVector{Float64}(nn_p);
        global nn_p = nn_params .* 0 + Float64(1e-4) * randn(rng, eltype(nn_params), size(nn_params));
        error = loss(nn_p);
    end;

    return nn_p;
end;

println("----Experiment Config Loaded----");