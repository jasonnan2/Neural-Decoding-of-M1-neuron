% load( 'ot.mat');
% load( 'nt.mat');
% load ('st.mat');
close all
error_mat=zeros(7,8,3);
for d=1:7
    ote=zeros(1,8);
    nte=ote;
    ste=ote;
    for r=1:5
        holder=orig_analysis.error(r,d);
        ote=ote+holder{1};
        holder=shift_analysis.error(r,d);
        ste=ste+holder{1};
        holder=noise_analysis.error(r,d);
        nte=nte+holder{1};
    end
    error_mat(d,:,1)=ote;
    error_mat(d,:,2)=ste;
    error_mat(d,:,3)=nte;
end
error_mat=error_mat*8/5;

figure
[time trial] = meshgrid(1:8);

mesh(error_mat(:,:,1),'edgecolor','r')
hold on
mesh(error_mat(:,:,2),'edgecolor','b')
hold on
mesh(error_mat(:,:,3),'edgecolor','g')
xlabel('Recording Time (ms)')
ylabel('Number of trials')
zlabel('Error (mm)')
title('Plot of Error Across Time and Trial Number')

set(gca,'Xtick',[1:8],'Xticklabel',[1:8]*20,'Yticklabel',[350:20:470])
legend({'Original';'Shifted';'Noise'},'location','best')
        
        
        
        
        