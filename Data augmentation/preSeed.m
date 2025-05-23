
function [train,randD] = preSeed(train_trial,D)
    N=91;K=8;
    Num_trials=350:20:470;
    
    % Choosing 20 rtandom neurons
    struct_hold=train_trial;
    clear train_trial;
    randD=randperm(97);
    for k=1:K
        for n=1:N
            data=struct_hold(n,k).spikes;
            pos=struct_hold(n,k).handPos;
            for d=1:D
                train_trial(n,k).spikes(d,:)=data(randD(d),:);
            end
            train_trial(n,k).handPos=pos;
        end
    end
    
    
    
    
    
    
    randN = randperm(N);
    randK = randperm(K);
    counter=1;
    for n = 1:N
        for k = 1:K
            reshapeSpike(counter).spikes = train_trial(randN(n),randK(k)).spikes;
            reshapeArm(counter).handPos = train_trial(randN(n),randK(k)).handPos;
            counter=counter+1;
        end
    end
    for r = 1:5
        randP = randperm(N*K);
        for i = 1:length(Num_trials) 
            for j = 1:Num_trials(i)
                train(r,i).data(j).spikes = reshapeSpike(1,randP(j)).spikes;
                train(r,i).data(j).handPos = reshapeArm(1,randP(j)).handPos;
            end
        end
    end
end
        