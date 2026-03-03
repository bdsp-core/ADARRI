function [ClasA,ClasB,ClasC,ClasAB,ClasBC]=fcn_Sunils_data(interval,itimes)

interval=interval*1000; itimes=itimes*1000; adRRI=[0 (abs(diff(interval)))];

% A = Brandons method
theta1=10; [RlocA]=fcn_calculate_absdiff(adRRI,theta1);
Ay=fcn_Flag_Identification(interval,adRRI,theta1);
ClasA=ones(size(RlocA)); ClasA(1,find(RlocA==0 | Ay==0))=0;

% B = Berntsons method
MEDn=iqr(adRRI)/2*3.32; MADa=((median(adRRI))-2.9*iqr(adRRI))/3;
t1=(abs(MEDn)+abs(MADa))/2; % Threshold for Berntson method
ClasB=fcn_Flag_Identification(interval,adRRI,t1);

% C = Cliffords method
[hrv] = fcn_clean_hrv4([itimes' interval']);
ClasC=zeros(1,length(itimes)); l=1;
for k=1:length(itimes)
    if length(hrv)==l;
        break
    elseif itimes(1,k)==hrv(l,1)
        l=l+1;
    else
        ClasC(1,k)=1;
    end
end

% A and B combined
ClasBC=zeros(1,size(ClasA,2)); ClasBC(1,find(ClasA==1 | ClasB==1))=1;

% B and C combined
ClasBC=zeros(1,size(ClasA,2)); ClasBC(1,find(ClasB==1 | ClasC==1))=1;

