function [se,sp,acc]=fcnGetSensSpecRRI(x,th,y)

%% get sensitivity and specificity for a given value of th
th0=th;
th1=th;

se=zeros(length(th),length(th));
sp=zeros(length(th),length(th));
acc=zeros(length(th),length(th));
for i=1:length(th0);
    for k=i:length(th1);
        for j=1:length(x)
            data=x{1,j};
            
            if min(data)<exp(th0(i));
                yh=1; % Data lower than theta0 is artifact
            elseif max(data)>exp(th1(k));
                yh=1; % Data above theta1 is artifact
            else yh=0; % Rest is not an artifact
            end
            
            if yh==1
                yyh(j,1)=1;
            else
                yyh(j,1)=0;
            end
        end
        se(i,k) = sum(y==1 & yyh==1)/sum(y==1);
        sp(i,k) = sum(y==0 & yyh==0)/sum(y==0);
        acc(i,k)= sum(y==yyh)/length(y);
    end
end

end