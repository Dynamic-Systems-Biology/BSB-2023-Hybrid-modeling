println("Number of Threads: $(Threads.nthreads())")

using Random

using Suppressor
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using BSON
using JSON
using TimerOutputs
using DelimitedFiles
using SciMLBase
using ComponentArrays
using Dates
using LineSearches
using JLSO

include("./util/CommandLineParser.jl");
include("./util/PrintLoggingUtil.jl");

using .CommandLineParser;

currentTime = Dates.format(now(), "ddmmyy_HHMMSS");
parsed_args = parseCommandline2();

folder_model = parsed_args["folder-model"];
base_folder = "$(pwd())/$(folder_model)";

increase_train_size_factor = parsed_args["increase-train-size-by"];

experimentConfig = JSON.parsefile("$(base_folder)/$(parsed_args["config"])");

condition_folder = "$(base_folder)/$(experimentConfig["conditionFolder"])";
results_folder = "$(condition_folder)/results_$(split(parsed_args["config"], ".")[1])-$(increase_train_size_factor)-$(Dates.format(now(), "ddmmyy_HHMM"))";
plots_folder = "$(results_folder)/plots";
val_plots_folder = "$(results_folder)/val_plots"
val_param_folder = "$(results_folder)/val_param"
loss_file_name = "$(base_folder)/$(experimentConfig["modelFile"])";


train_size = increase_train_size_factor * experimentConfig["trainSize"];
test_size = experimentConfig["testSize"];
val_size = experimentConfig["valSize"];
adam_maxiters = experimentConfig["adamIterations"];
bfgs_maxiters = experimentConfig["bfgsIterations"];
saveAt = experimentConfig["savePointsAt"];
plotScatter = experimentConfig["plotScatter"];
n_conditions = train_size + test_size + val_size;
increase_train_size_factor = nothing;

mkpath(results_folder);
mkpath(plots_folder);
mkpath(val_plots_folder);
mkpath(val_param_folder);
include(loss_file_name);

println("----ADAM [$(adam_maxiters)]----")
println("----BFGS [$(bfgs_maxiters)]----")

println("----Train      Size: $(train_size)----")
println("----Test       Size: $(test_size)----")
println("----Validation Size: $(val_size)----")

println("----Load initial conditions----");
conditions = Dict[];

for i in 1:train_size
    push!(conditions, BSON.load("$(condition_folder)/train_condition_$(i).bson"))
end;

for i in 1:val_size
    push!(conditions, BSON.load("$(condition_folder)/val_condition_$(i).bson"))
end;

for i in 1:test_size
    push!(conditions, BSON.load("$(condition_folder)/test_condition_$(i).bson"))
end;

println("----Conditions loaded----");
println("----config initial weights----");

neuralNetworkInitialParameters = @suppress_err begin
    config_initial_weights()
end;

println("----initial weights configured----");

initial_loss = loss(neuralNetworkInitialParameters)
println("----initial loss: $(initial_loss)----");

print_sumary(train_size, val_size, test_size, test_loss, conditions, neuralNetworkInitialParameters);

optimizationFunction = Optimization.OptimizationFunction(
    (x, p) -> loss(x),
    Optimization.AutoForwardDiff()
);

optimizationProblem = Optimization.OptimizationProblem(
    optimizationFunction,
    neuralNetworkInitialParameters
);

learningRates = [0.1, 0.05, 0.001];
iterations = [adam_maxiters, adam_maxiters, adam_maxiters];

neuralNetworkInitialParameters = nothing;
const timerOutput = TimerOutput();
res1 = nothing;

for i in 1:size(learningRates)[1]
    global res1 = @timeit timerOutput "ADAM_TRAINING" begin
        @suppress_err begin
            Optimization.solve(
                optimizationProblem,
                ADAM(learningRates[i]),
                maxiters=iterations[i],
                callback=callback,
                progress=false
            )
        end
    end

    validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters]
    print_sumary(train_size, val_size, test_size, test_loss, conditions, validationMinimizer)

    global optimizationProblem = Optimization.OptimizationProblem(
        optimizationFunction,
        validationMinimizer
    )

    global otp_extra_step = 100
end;

validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters];

optimizationProblem = Optimization.OptimizationProblem(
    optimizationFunction,
    validationMinimizer
);

JLSO.save("$(results_folder)/adam_parameters.jlso", :ude_parameters => res1.minimizer);

otp_extra_step = 20;

res2 = @timeit timerOutput "BFGS_TRAINING" begin
    @suppress_err begin
        Optimization.solve(
            optimizationProblem,
            BFGS(
                initial_stepnorm=0.01f0,
                linesearch=LineSearches.BackTracking()
            ),
            allow_f_increases=false,
            maxiters=bfgs_maxiters,
            callback=callback,
            progress=false
        )
    end
end;

JLSO.save("$(results_folder)/bfgs_parameters.jlso", :ude_parameters => res2.minimizer);

validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters];

print_sumary(train_size, val_size, test_size, test_loss, conditions, validationMinimizer);

show(timerOutput)
println();

open("$(results_folder)/times.json", "a") do io
    JSON.print(io, TimerOutputs.todict(timerOutput), 4)
end;

include("./util/PlotUtil.jl");

println("\n----Ploting Conditions----");

plot_all_condtions(res1.minimizer, res2.minimizer);

validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters];

plot_all_condtions(validationMinimizer);

println("\n----Ploting Lossess----");

valLosses = readdlm("$(results_folder)/valLoss.csv", Float32);
testLosses = readdlm("$(results_folder)/testLoss.csv", Float32);
trainLosses = readdlm("$(results_folder)/trainLoss.csv", Float32);

minPoint = (min_val_loss_i, min_val_loss_v);

println("MinPoint: $(minPoint)");

figure = plot_losses(trainLosses, valLosses, minPoint);
savefig(figure, "$(plots_folder)/losses_.svg");

figure = plot_losses(trainLosses, valLosses, testLosses, minPoint);
savefig(figure, "$(plots_folder)/all_losses_.svg");
