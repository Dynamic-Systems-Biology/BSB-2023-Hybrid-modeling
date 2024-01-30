function print_sumary(n_train, n_val, n_test, test_loss_func, conditions, parameter)

    println("\n\n-------TRAIN SET--------");
    
    losses = zeros(n_train + n_test + n_val);

    for i in 1:n_train
        losses[i] = test_loss_func(parameter, conditions[i]);
        print("\tCondition $(i): $(losses[i])\n");
    end;

    println("-------VAL SET--------");

    for i in 1:n_val
        index = n_train + i;
        losses[index] = test_loss_func(parameter, conditions[index]);
        print("\tCondition $(i): $(losses[index])\n");
    end;

    println("-------TEST SET--------");

    for i in 1:n_test
        index = n_train + n_val + i;
        losses[index] = test_loss_func(parameter, conditions[index]);
        print("\tCondition $(i): $(losses[index])\n");
    end;

    println("---------------\n");

    open("$(results_folder)/loss_per_condition.csv", "a+") do io
        writedlm(io, losses', ',');
    end;

    losses = nothing;
end;
