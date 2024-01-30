module CommandLineParser
using ArgParse

export parseCommandline, parseCommandline2

function parseCommandline2()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--folder-model"
            help = "folder in witch the model and json experiment config is stored"
            required = true
        "--config"
            help = "Pass the json file with experiment config"
            required = true
        "--increase-train-size-by"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

function parseCommandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--folder-model"
            help = "folder in witch the model is stored"
            required = true
        "--condition-folder"
            help = "folder in which the conditions are saved in BSON format"
            arg_type = String
            default = "conditions"
        "--results-folder"
            help = "folder in which the results will be saved"
            arg_type = String
            default = "results"
        "--loss-file-name"
            help = "file that has the loss function that must be optimized"
            arg_type = String
            default = "loss_function.jl"
        "--condition-name-prefix"
            help = "the prefix of the file name with the conditions"
            arg_type = String
            default = "condition"
        "--train-size"
            help = "how many conditions should be used to train"
            arg_type = Int
            required = true
        "--test-size"
            help = "how many conditions should be used to test"
            arg_type = Int
            default = 0
        "--n-epochs"
            help = "how many epochs to be run"
            arg_type = Int
            default = 1
        "--ad-maxiters"
            help = "maxiters for ADAM optimizer"
            arg_type = Int
            default = 5000
        "--bs-maxiters"
            help = "maxiters for BFGS optimizer"
            arg_type = Int
            default = 2500
    end
    return parse_args(s)
end
end