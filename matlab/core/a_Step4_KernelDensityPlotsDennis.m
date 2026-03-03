clear all; clc; format compact;
% cd('F:\For Brandon\Artifact_Reduction')

%% create some data
% [X,Y] = fcnRealSampleData;

load('XY_RealData');
load('THETAS.mat');

%% look at density estimates for the plots

for i=31:50%length(X); % loop over patients
    tic
    % RRI
    data0=[];
    data1=[];
    y=Y{i}; x=X{i};
    for j=1:length(x);
        if  y(j)==1;
            data1 = [data1 x{1,j}]; % epochs of artifacts
        else
            data0 = [data0 x{1,j}]; % epochs of normal data
        end
    end
    limit=[min(log(data0)) max(log(data0)) min(log(data1)) max(log(data1))];
    xx=linspace(min(limit),max(limit),1000);
    
    % plot distributions for patient i
    figure(1); clf;
    subplot(321);
    fcnPlotKernelPDFestimatesRRI(xx,data0,data1);
    title 'Kernel PDF estimates RRI'
    
    % find ROC curve for patient i, and find optimal threshold [just use one threshold in this example, th]
    th = linspace(min(xx),max(xx),500);
    [se,sp,acc]=fcnGetSensSpecRRI(x,th,y);
    %
    % get index of best threshold that gives best accuracy
    [maxacc,ind] = max(acc(:));
    [m,n] = ind2sub(size(acc),ind);
    th0Best = th(m); % theta0, lower threshold
    th1Best = th(n); % theta1, upper threshold
    hold on; plot([th1Best th1Best],[-14 0],'k--'); plot([th0Best th0Best],[-14 0],'k--')
    
    subplot(323);
    plot(th,se(m,:),'b',th,sp(:,n),'r'); hold on;
    grid on; plot([th1Best th1Best],[0 1],'k--'); plot([th0Best th0Best],[0 1],'k--')
    xlabel('\theta'); title('Sensitivity (blue), specificity (red)');
    
    subplot(325); plot(th,acc(),'b'); hold on; % accuracy vs threshold
    grid on; plot([th1Best th1Best],[0 1],'k--'); plot([th0Best th0Best],[0 1],'k--')
    xlabel('\theta'); ylabel('Accuracy'); title('Accuracy vs threshold');
    drawnow;
    
    % Abs(Diff(RRI))
    data0=[];
    data1=[];
    y=Y{i}; x=X{i};
    for j=1:length(x);
        if  y(j)==1;
            data1 = [data1 x{2,j}]; % epochs of artifacts
        else
            data0 = [data0 x{2,j}]; % epochs of normal data
        end
    end
    limit=[min(log(data0)) max(log(data0)) min(log(data1)) max(log(data1))];
    xx=linspace(min(limit),max(limit),1000);
    
    % plot distributions for patient i
    figure(1);
    subplot(322); title 'Kernel PDF estimates abs(diff(RRI))'
    
    [maxvalue,f0,f1]=fcnPlotKernelPDFestimatesAbsDif(xx,data0,data1);
    
    
    % find ROC curve for patient i, and find optimal threshold [just use one threshold in this example, th]
    th = linspace(min(xx),max(xx),500);
    for j=1:length(th);
        [se2(j),sp2(j),acc2(j)]=fcnGetSensSpecAbsDif(x,th(j),y);
    end
    [~,jj]=max(acc2); % get index of best threshold that gives best accuracy
    thBest = th(jj);
    hold on; plot([thBest thBest],[0 maxvalue],'k--')
    
    subplot(324); plot(th,se2,th,sp2); % ROC curve (different style than usual, but same concept)
    grid on; hold on; plot([thBest thBest],[0 1],'k--')
    xlabel('\theta'); title('Sensitivity (blue), specificity (red)');
    subplot(326); plot(th,acc2); % accuracy vs threshold
    grid on; hold on; plot([thBest thBest],[0 1],'k--')
    xlabel('\theta'); ylabel('Accuracy'); title('Accuracy vs threshold');
    drawnow;
    
    THETAS(i,:)=[exp(th0Best); exp(th1Best); exp(thBest)];
    disp(['Patient ' num2str(i)]);
    save THETAS THETAS
    
    filename={'DetectedArtifacts',num2str(i),'.png'};
    filename=strjoin(filename,'');
    saveas(gcf,filename)
    toc
    %     g = input('enter to see next case');
end
disp('Done')


