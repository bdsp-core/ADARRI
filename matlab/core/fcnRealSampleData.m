function [X,Y]=fcnRealSampleData
mat_files=dir('SegmentScores_Artifact_*');

X=[];Y=[];
for pt=1:length(mat_files)
    load (mat_files(pt).name);
    
    %% Take only first 60 epochs
    while length(ECG0)>60
        ECG0{1,end}=[];
        ECG0(:,find(all(cellfun(@isempty,ECG0),1))) = []; %delete whole empty column
    end
    while length(ECG1)>60
        ECG1{1,end}=[];
        ECG1(:,find(all(cellfun(@isempty,ECG1),1))) = []; %delete whole empty column
    end
    Labels=[zeros(length(ECG0),1); ones(length(ECG1),1)];
    
    %% Calculate RRI from ECG
    data=[];
    for j=1:length(ECG0)
        epoch=ECG0{j};
        [interval,itimes,~,Rloc] = fcn_get_rr_interval(epoch,240);
        tt=itimes(1):0.01:itimes(end);
        yy=spline(itimes,interval,tt)*1000;
        Y0=abs(diff(yy));
        data{1,j}=yy; % RRI
        data{2,j}=Y0; % Abs Diff RRI
    end
    for i=1:length(ECG1)
        epoch=ECG1{i};
        [interval,itimes,~,Rloc] = fcn_get_rr_interval(epoch,240);
        tt=itimes(1):0.01:itimes(end);
        yy=spline(itimes,interval,tt)*1000;
        Y1=abs(diff(yy));
        data{1,i+j}=yy; % RRI
        data{2,i+j}=Y1; % Abs Diff RRI
    end
    
    X{pt}=data; Y{pt}=Labels; % Data for this patient
end

end