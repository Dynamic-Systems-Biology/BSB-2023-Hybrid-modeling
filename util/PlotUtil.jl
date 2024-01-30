using Plots

ENV["GKSwstype"] = 100;

function plot_all_condtions(adam_minimizer, bfgs_minimizer)
    for i in 1:n_conditions
        print("\tCondition $(i)\n");
    
        if i <= train_size
            type  = "Training";
            index = i;
        elseif i <= train_size + val_size  
            type  = "Validation";
            index =  i - train_size;
        else
            type = "Test";
            index =  i - (train_size + val_size);
        end;
    
        condition = conditions[i];
        solution = predict(adam_minimizer, condition[:u0]);
        X̂ = Array(solution);
        l = mae(X̂, condition[:X])
        
        figure = plot_experiment(
            plotScatter, 
            solution.t, 
            X̂,
            condition,
            "ADAM results (condition $(index), $(type)) [$(l)]", 
            species_name, 
            colors
        );

        savefig(figure, "$(plots_folder)/$(type)_adam_condition_$(index).svg");

        condition = conditions[i];
        solution = predict(bfgs_minimizer, condition[:u0]);
        X̂ = Array(solution);
        l = mae(X̂, condition[:X])

        figure = plot_experiment(
            plotScatter, 
            solution.t, 
            X̂, 
            condition, 
            "BFGS results (condition $(index), $(type))", 
            species_name, 
            colors
        );
        savefig(figure, "$(plots_folder)/$(type)_bfgs_condition_$(index).svg");
    end;    
end;

function plot_all_condtions(minimizer)
    for i in 1:n_conditions
        print("\tCondition $(i)\n");

        if i <= train_size
            type  = "Training";
            index = i;
        elseif i <= train_size + val_size  
            type  = "Validation";
            index =  i - train_size;
        else
            type = "Test";
            index =  i - (train_size + val_size);
        end;

        condition = conditions[i];
        solution = predict(minimizer, condition[:u0]);
        X̂ = Array(solution);
        l = mae(X̂, condition[:X])

        figure = plot_experiment(
            plotScatter,
            solution.t,
            X̂,
            condition,
            "$(type) - Initial Condition $(index) [$(l)]", 
            species_name,
            colors
        );
    
        savefig(figure, "$(val_plots_folder)/$(type)_condition_$(index).svg");
    end;    
end


function plot_losses(losses, valLosses, testLosses, minPoint)

    default(size = (900, 500));
    N = size(valLosses)[1];

    plot(
        1:N, losses, 
        yaxis = :log10, xaxis = :log10, 
        label = "Training", color = :red,
        title = "Loss Function"
    );

    plot!(
        1:N, valLosses, 
        yaxis = :log10, xaxis = :log10, 
        label = "Validation", color = :blue,
    );

    annotate!(minPoint[1], minPoint[2]+ 10, text("$(minPoint[2])", :black, :right, 6))
    plot!([minPoint[1]], label="Best Parameter", color = :gray, seriestype = :vline);

    plot_loss_function = plot!(
        1:N, testLosses, 
        yaxis = :log10, xaxis = :log10, 
        label = "Test", color = :black,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm
    );

    xlabel!("Iterations")
    ylabel!("Loss")

    return plot_loss_function;
end;

function plot_losses(losses, valLosses, minPoint)

    default(size = (900, 500));
    N = size(losses)[1];

    plot(
        1:N, losses, 
        yaxis = :log10, xaxis = :log10, 
        label = "Training", color = :red,
        title = "Loss Function"
    );

    xlabel!("Iterations")
    ylabel!("Loss")

    annotate!(minPoint[1], minPoint[2] + 10, text("$(minPoint[2])", :black, :right, 6))
    plot!([minPoint[1]], label="Best Parameter", color = :gray, seriestype = :vline);

    plot_loss_function = plot!(
        1:N, 
        valLosses, 
        yaxis = :log10, xaxis = :log10, 
        label = "Validation", color = :blue,
    );

    return plot_loss_function;
end;

function plot_losses(losses)

    default(size = (900, 500));

    plot_loss_function = plot(
        1:size(losses)[1], 
        losses,
        yaxis = :log10,
        xaxis = :log10, 
        color = :blue,
        title = "Loss Function",
    );
    
    xlabel!("Iterations")
    ylabel!("Loss")
    
    return plot_loss_function;
end;

function plot_experiment(scatter, t, predicted, condition, title, species_name, colors)

    default(size = (900, 500))

    if !scatter
        plot(
            t, 
            condition[:X]',
            title = title,
            alpha=0.75, 
            labels=permutedims(species_name),
            color=permutedims(colors),
            legend=:outertopright
        );
    else
        plot(
            t, 
            condition[:X]',
            title = title,
            alpha=0.75, 
            labels=permutedims(species_name),
            color=permutedims(colors),
            seriestype=:scatter,
            legend=:outertopright
        );
    end;

    xlabel!("t");
    ylabel!("concentration");

    experiment_plot = plot!(
        t, 
        predicted',
        alpha=0.75,
        labels=permutedims(species_name),
        color=permutedims(colors),
        ls=:dash,
        left_margin=5Plots.mm
    );

    return experiment_plot;
end;