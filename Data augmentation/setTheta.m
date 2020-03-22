

% Function to define theta after shifting 

function [Parameters] = setTheta(train, T)
% FIX NORMALIZATION
N=91;K=8;
D=size(train(1,1).data(1,1).spikes,1);
Num_trials=350:20:470;
for r = 1:5
    for i = 1:length(Num_trials)
        T_max(r,i) = max(T{r,i});
        T_sum_sum(r,i) = sum(sum(T{r,i}));
        Parameters(r,i).A=zeros(4,4);
        top=zeros(4,4);
        bottom=top;
        for j = 1:Num_trials(i)   
            for t= 2:T{r,i}(j)
                zt = train(r,i).data(j,1).arm(:,t);
                zt1 =  train(r,i).data(j,1).arm(:,t-1);
                top = top+zt*zt1';
                bottom = bottom+zt1*zt1';
            end
        end
        Parameters(r,i).A=Parameters(r,i).A+top*inv(bottom);
    end
end

for r=1:5
    for i=1:length(Num_trials)
        Parameters(r,i).Q=zeros(4,4);
        for j = 1:Num_trials(i)
            holder=zeros(4,4);
            for t=2:T{r,i}(j)
                zt= train(r,i).data(j,1).arm(:,t);
                zt1= train(r,i).data(j,1).arm(:,t-1);
                
                holder=holder+(zt-Parameters(r,i).A*zt1)*(zt-Parameters(r,i).A*zt1)';
            end
            Parameters(r,i).Q=Parameters(r,i).Q+inv(T{r,i}(j)-1)*holder*T{r,i}(j)/T_sum_sum(r,i);
        end
    end
end

for r=1:5
    for i=1:length(Num_trials)
        Parameters(r,i).C=zeros(D,4); 
        holder=zeros(D,4);
        top = holder; 
        bottom=zeros(4,4);
        for j = 1:Num_trials(i)
            for t = 1:T{r,i}(j)
                xt= train(r,i).data(j,1).spikes(:,t);
                zt= train(r,i).data(j,1).arm(:,t);
                top=top+xt*zt';
                bottom=bottom+zt*zt';
            end            
        end
        C=(top*inv(bottom));
        C(C==0)=0.001;
        Parameters(r,i).C=C;
    end
end

for r=1:5
    for i=1:length(Num_trials)
        Parameters(r,i).R=zeros(D,D);
        
        for j = 1:Num_trials(i)
            holder=zeros(D,D);
            for t = 1:T{r,i}(j)
                xt= train(r,i).data(j,1).spikes(:,t);
                zt= train(r,i).data(j,1).arm(:,t);
                holder=holder+(xt-Parameters(r,i).C*zt)*(xt-Parameters(r,i).C*zt)';
            end
            Parameters(r,i).R=Parameters(r,i).R+inv(T{r,i}(j))*holder*T{r,i}(j)/T_sum_sum(r,i);
        end
    end
end
           

for r=1:5
    for i=1:length(Num_trials)
        Parameters(r,i).init=[];
        for j = 1:Num_trials(i)
            Parameters(r,i).init=[Parameters(r,i).init,train(r,i).data(j,1).arm(:,1)];
        end
        Parameters(r,i).Zi=mean(Parameters(r,i).init,2);
        Parameters(r,i).V=cov(Parameters(r,i).init');
    end
end

end


