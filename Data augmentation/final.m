    
%% Final Project

for looper=1:3
    clear -looper
    load hw4.mat
    num_neurons=[60 80 97];
    D=num_neurons(looper);
    %% Seeding trials
    [train,randD] = preSeed(train_trial,D);

    %% Processing shifting data augmentation
    [shift,T]=data_shift(train);
    [noise,T_noise]=data_noise(train);
    %% processing testing
    [test,T_test]=test_seed(test_trial,randD,D);
    %% Training
    theta_orig=setTheta(shift,T);
    theta_noise=setThetaAug(noise,T_noise);
    theta_shift=setThetaAug(shift,T);
    %% Testing
    orig_analysis=testing(test,T_test,theta_orig);
    shift_analysis=testing(test,T_test,theta_shift);
    noise_analysis=testing(test,T_test,theta_noise);

    %% Plotting 
    figure
    hold on
    oe=orig_analysis.error;
    oe(oe==NaN)=0;
    se=shift_analysis.error;
    se(se==NaN)=0;
    ne=noise_analysis.error;
    ne(ne==NaN)=0;
    plot(oe,'k')
    plot(se,'b')
    plot(ne,'r')
    legend({'original';'shifted';'noise'},'location','best')
    xlabel('Number of Trials used')
    ylabel('Error')
    title('Model Performance with Data Augmentation, D=20')
    set(gca,'XTick',[1 2 3 4 5 6 7],'XTicklabel',[350:20:470])

    a(looper).mean=[mean(oe),mean(ne),mean(se)];
    a(looper).variance=[mean(orig_analysis.variance) mean(noise_analysis.variance) mean(shift_analysis.variance)];
end

a








