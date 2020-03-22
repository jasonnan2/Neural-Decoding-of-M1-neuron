function [test,T_test]=test_seed(test_trial,randD,D)
N=91;K=8;bin=20;
T_min_test=ones(1,K)*100;
T_test=zeros(N,K);

% Choosing 20 rtandom neurons
    struct_hold=test_trial;
    clear test_trial;
    for k=1:K
        for n=1:N
            data=struct_hold(n,k).spikes;
            pos=struct_hold(n,k).handPos;
            for d=1:D
                test_trial(n,k).spikes(d,:)=data(randD(d),:);
            end
            test_trial(n,k).handPos=pos;
        end
    end
    

for n=1:N
    for k=1:K
        test_data=test_trial(n,k).spikes;
        test_arm=test_trial(n,k).handPos;
        T_test(n,k)=floor(size(test_data,2)/bin);
        if T_test(n,k)<T_min_test(k)
            T_min_test(k)=T_test(n,k);
        end
        for t=1:T_test(n,k)
            range=test_data(:,(t-1)*20+1:20*t);
            test(n,k).spikes(:,t)=sum(range,2);
            test(n,k).arm(1,t)=test_arm(1,20*t);
            test(n,k).arm(2,t)=test_arm(2,20*t);
            if t==1
                test(n,k).arm(3,t)=(test_arm(1,20)-test_arm(1,1))/20;
                test(n,k).arm(4,t)=(test_arm(2,20)-test_arm(2,1))/20;
            else 
                %train(n,k).arm(3,t)=(train_arm(1,20*t)-train_arm(1,(t-1)*20+1))/20;
                %train(n,k).arm(4,t)=(train_arm(2,20*t)-train_arm(2,(t-1)*20+1))/20;
                test(n,k).arm(3,t)=(test(n,k).arm(1,t)-test(n,k).arm(1,t-1))/20;
                test(n,k).arm(4,t)=(test(n,k).arm(2,t)-test(n,k).arm(2,t-1))/20;
            end
        end
        test(n,k).data=sum(test(n,k).spikes);
    end
end
end

