function analysis=testing(test,T_test,theta)
error=[]; N=91;K=8;D=20;

%% Testing trial
for r=1:5
    for d=1:7
        A=theta(r,d).A;
        Q=theta(r,d).Q;
        C=theta(r,d).C;
        C(C==0)=0.001;
        R=theta(r,d).R;
        R(R==0)=0.001;
        Zi=theta(r,d).Zi;
        V=theta(r,d).V;
        V(V==0)=0.001;
        for k=1:K
            for n=1:N
                mu_update=zeros(4,T_test(n,k));
                mu_update(:,1)=Zi;
                sigma_update=zeros(4,4,T_test(n,k));
                sigma_update(:,:,1)=(V);
                prediction(n,k).state(:,1)=Zi;
                prediction(n,k).cov(:,:,1)=V;
                for t=2:T_test(n,k)
                    x=test(n,k).spikes(:,t);
                    %One step prediction
                    mu_update(:,t)=A*mu_update(:,t-1);
                    sigma_update(:,:,t)=A*sigma_update(:,:,t-1)*A'+Q;
                    sig=sigma_update(:,:,t);
                    sig(sig==0)=0.001;
                    % Measurement update
                    fix=C*sig*C'+R;
                    fix(fix==0)=0.001;
                    kalman=sig*C'*inv(fix);
                    guess=mu_update(:,t)+kalman*(x-C*mu_update(:,t));
                    prediction(n,k).state(:,t)=guess;
                    mu_update(:,t)=guess;
                    covariance=sig-kalman*C*sig;
                    prediction(n,k).cov(:,:,t)=covariance;
                    sigma_update(:,:,t)=covariance;
                end
            end
        end
        
        %% error calculation
        min_bins=min(min(T_test));
        total_error=0;
        for k=1:K
            for n=1:N
                x_real=test(n,k).arm(1,1:min_bins);
                y_real=test(n,k).arm(2,1:min_bins);
                x_predict=prediction(n,k).state(1,1:min_bins);
                y_predict=prediction(n,k).state(2,1:min_bins);
                total_error=total_error+(abs(x_real-x_predict)+abs(y_real-y_predict))/min_bins;
            end
        end
        error(r,d)=sum(total_error)/(N*K);
        
    end
end
analysis.error=mean(error);
analysis.variance=var(error);
end
