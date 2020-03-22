N=91;K=8;D=97;bin=20;
% Processing training data
Num_trials=[100 200 300 400 500 600 700];
for r=1:5
    for i=1:7
        T{r,i}=zeros(1,Num_trials(i));
        for j=1:Num_trials(i)
            % extract individual trials
            train_data=train(r,i).data(j).spikes;
            train_arm=train(r,i).data(j).handPos;

            % Truncating the dataset
            s=length(train_data); truncated_length=s-mod(s,bin)-bin;
            train_data_hold=train_data(1:truncated_length);
            train_arm_hold=train_arm(1:truncated_length);
            T{r,i}(j)=(truncated_length/bin)-1;

%             for b=0:bin-1
%                 for t=1:T{i}(j)
%                     % setting binning ranges
%                     low=(t-1)*20+1+b;
%                     up=20*t+b;
%                     range=train_data_hold(:,low:up);
%                     shift(r,i).data(j,b+1).spikes=sum(range,2);
%                     shift(r,i).data(j,b+1).arm(1,t)=train_arm_hold(1,up);
%                     shift(r,i).data(j,b+1).arm(2,t)=train_arm_hold(2,up);
%                     if t==1
%                         shift(r,i).data(j,b+1).arm(3,t)=(train_arm_hold(1,up)-train_arm_hold(1,low))/20;
%                         shift(r,i).data(j,b+1).arm(4,t)=(train_arm_hold(2,up)-train_arm_hold(2,low))/20;
%                     else
%                         shift(r,i).data(j,b+1).arm(3,t)=(shift(r,i).data(j,b+1).arm(1,t)-shift(r,i).data(j,b+1).arm(1,t-1))/20;
%                         shift(r,i).data(j,b+1).arm(4,t)=(shift(r,i).data(j,b+1).arm(2,t)-shift(r,i).data(j,b+1).arm(2,t-1))/20;
%                     end
%                 end
%             end
        end
    end
end