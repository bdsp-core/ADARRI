%%
clear; clc; clf;
cd('D:\Dennis_Matlab\Artifact_reduction\Selected_files'); % folder to start in
pd = pwd;
mat_files=dir('*.mat');

for k=42%:length(mat_files) % loop over all patients
    cd('D:\Dennis_Matlab\Artifact_reduction\Selected_files');
    
    filename=mat_files(k).name;
    load (filename);
    kernstep=0:0.01:1;
    
    % Calculate parameters H0
    var=ECG0;
    tempYY0=[]; % collect all data for patient k in this vector
    tempRR0=[];
    for i=1:length(var) % loop over all epochs for patient k
        try epoch=var{1,i};
            cd ('D:\Dennis_Matlab\Artifact_reduction');
            [interval,itimes,~,Rloc1] = fcn_get_rr_interval(epoch,240);
            
            rri=interval*1000;   t=itimes;
            tt=t(1):0.01:t(end);
            yy=spline(t,rri,tt);
            Y0=abs(diff(yy));
        catch
            disp('for some reason there is an error....')
        end
        tempYY0=[tempYY0 Y0];
        tempRR0=[tempRR0 yy];
    end
    YY0{k}=tempYY0;
    RR0{k}=tempRR0;
    
    % Calculate parameters H1
    var=ECG1;
    tempYY1=[]; % collect all data for patient k in this vector
    tempRR1=[];
    for i=1:length(var) % loop over all epochs for patient k
        try epoch=var{1,i};
            cd ('D:\Dennis_Matlab\Artifact_reduction');
            [interval,itimes,~,Rloc1] = fcn_get_rr_interval(epoch,240);
            cd(pd);
            
            rri=interval*1000;   t=itimes;
            tt=t(1):0.01:t(end);
            yy=spline(t,rri,tt);
            Y1=abs(diff(yy));
            
        catch
            disp('for some reason there is an error....')
        end
        tempYY1=[tempYY1 Y1];
        tempRR1=[tempRR1 yy];
    end
    YY1{k}=tempYY1;
    RR1{k}=tempRR1;
    
    %% Plot histograms for abs(diff( of RR intervals ))
    figure(1); clf;
    subplot(211);
    limit=[min(log(YY0{k})) max(log(YY0{k})) min(log(YY1{k})) max(log(YY1{k}))];
    xx=linspace(min(limit)-1,max(limit)+1,1000);
    f0=hist(log(YY0{k}),xx); f0=f0/sum(f0);
    f1=hist(log(YY1{k}),xx); f1=f1/sum(f1);
    
    cd ('D:\Dennis_Matlab\Artifact_reduction');
    plot(xx,(f0),'b');
    hold on
    plot(xx,(f1),'r');
    
    cd ('D:\Dennis_Matlab\Artifact_reduction');
    hh0 = createPatches(xx,(f0),.01,'b',.2);
    hold on
    hh1 = createPatches(xx,(f1),.01,'r',.2);
    
    xlabel 'Time difference of RR interval in log(ms)'
    ylabel 'Log of Probability Density Function'
    title('Log of absolute value of difference in RR interval');
    legend ('Normal data','Data contains artifact')
    xlim ([min(limit) max(limit)])
    hold off
    
    % Plot histograms for log of RR intervals
    figure(1);
    subplot(212);
    limit=[abs(min(log(RR0{k}))) abs(max(log(RR0{k}))) abs(min(log(RR1{k}))) abs(max(log(RR1{k})))];
    xx=linspace(min(limit)-1,max(limit)+1,1000);
    f0=hist(log(RR0{k}),xx); f0=f0/sum(f0);
    f1=hist(log(RR1{k}),xx); f1=f1/sum(f1);
    
    cd ('D:\Dennis_Matlab\Artifact_reduction');
    plot(xx,log(f0),'b');
    hold on
    plot(xx,log(f1),'r');
    
    cd ('D:\Dennis_Matlab\Artifact_reduction');
    hh0 = createPatches(xx,log(f0),.01,'b',.2);
    hold on
    hh1 = createPatches(xx,log(f1),.01,'r',.2);
    
    xlabel 'Time of RR interval in log(ms)'
    ylabel 'Log of the Log of Probability Density Function'
    title('Log of RR interval');
    legend ('Normal data','Data contains artifact')
    xlim ([min(limit) max(limit)])
    hold off
    
    cd('D:\Dennis_Matlab\Artifact_reduction\Figures')
    filename={filename(1:end-4),'.png'};
    filename=strjoin(filename,'');
    saveas(gcf,filename)
end



