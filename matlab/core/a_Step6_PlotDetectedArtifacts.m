clear all; clc; format compact;
%cd('F:\For Brandon\Artifact_Reduction')
% cd('R:\Dennis_ECG_ArtifactReduction'); 
cd('J:\Dropbox (Partners HealthCare)\0_Work\Administrative\Mentoring_And_Colleagues\RebergenDennis\Code\Dennis_ECG_ArtifactReduction'); 
mat_files=dir('*.mat');
load('THETAS.mat');

%%
for k=10%:10%:length(mat_files) % loop over all patients
    filename=mat_files(k).name;
    load (filename);
    
    th0=THETAS(k,1); %theta0 of RRI
    th1=THETAS(k,2); %theta1 of RRI
    th=THETAS(k,3); %theta for abs(diff(RRI)
    % Take only first 60 epochs
    while length(ECG0)>60
        ECG0{1,end}=[];
        ECG0(:,find(all(cellfun(@isempty,ECG0),1))) = []; %delete whole empty column
    end
    while length(ECG1)>60
        ECG1{1,end}=[];
        ECG1(:,find(all(cellfun(@isempty,ECG1),1))) = []; %delete whole empty column
    end
    
    % Calculate parameters H0
    for m=1:2
        if m==2
            var=ECG0;
        else var=ECG1;
        end
        for i=1:length(var) % loop over all epochs for patient k
            try epoch=var{1,i};
                time=(1:length(epoch))/240*1000;
                [interval,itimes,~,Rloc1] = fcn_get_rr_interval(epoch,240);
                rri=interval*1000; t=itimes.*1000;
                tt=t(1):10:t(end);
                yy=spline(t,rri,tt);
                Y0=abs(diff(yy));
                Y00=[0 resample(Y0,length(rri)-1,length(Y0))];
                % Divide Rloc in artifact and normal
                Rloc_rri=zeros(1,length(rri));
                for n=1:length(rri)
                    if rri(n)<th0
                        Rloc_rri(1,n)=1;
                    elseif rri(n)>th1
                        Rloc_rri(1,n)=1;
                    end
                end
                Rloc_rri_normal= find(Rloc_rri==0);
                Rloc_rri_artifact=find(Rloc_rri==1);
                
                Rloc_abs=zeros(1,length(Y00));
                for n=1:length(Y00)
                    if Y00(n)>th
                        Rloc_abs(1,n)=1;
                    end
                end
                Rloc_abs_normal= find(Rloc_abs==0);
                Rloc_abs_artifact=find(Rloc_abs==1);
                
                % plot
                clf;
                figure(1),
                % RRI
                ha(1)=subplot(411);
                plot(time,epoch); hold on;
                plot(time(Rloc1(Rloc_rri_normal)),epoch(Rloc1(Rloc_rri_normal))+30,'gr*')
                plot(time(Rloc1(Rloc_rri_artifact)),epoch(Rloc1(Rloc_rri_artifact))+30,'r*')
                title 'Artifact detection based on RRI'
                xlabel 'Time in ms'
                ylabel 'Amplitude'
                
                ha(2)=subplot(412);
                plot(t,rri,'.'); hold on; 
                plot([t(1),t(end)],[th0,th0],'k--')
                plot([t(1),t(end)],[th1,th1],'k--')
                title 'RRI of Artifact detection based on RRI'
                xlabel 'Time in ms'
                ylabel 'RRI in ms'
                
                % Abs(diff(RRI)
                ha(3)=subplot(413);
                plot(time,epoch); hold on;
                plot(time(Rloc1(Rloc_abs_normal)),epoch(Rloc1(Rloc_abs_normal))+30,'gr*')
                plot(time(Rloc1(Rloc_abs_artifact)),epoch(Rloc1(Rloc_abs_artifact))+30,'r*')
                title 'Artifact detection based on abs(diff(RRI))'
                xlabel 'Time in ms'
                ylabel 'Amplitude'
                
                ha(4)=subplot(414);
                plot(t(1:end),Y00,'.'); hold on; 
                plot([tt(1),tt(end)],[th,th],'k--')
                title 'abs(diff(RRI)) of Artifact detection based on RRI'
                xlabel 'Time in ms'
                ylabel 'abs(diff(RRI)) in ms'
                linkaxes(ha, 'x');      % Link all axes in x
                
            catch
                disp('for some reason there is an error....')
            end
            disp([m i])
            g=input('Press enter for next');
        end
    end
end