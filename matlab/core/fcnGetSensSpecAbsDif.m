function [se,sp,acc]=fcnGetSensSpecAbsDif(x,th,y); 

%% get sensitivity and specificity for a given value of th

yh = zeros(size(y)); 
for j=1:length(x)        
   data=x{2,j};  
   if max(data)>exp(th); 
       yh(j)=1; % classify as epoch with artifact
   else
       yh(j)=0; % classify as epoch with no artifact
   end
end

se = sum(y==1 & yh==1)/sum(y==1); 
sp = sum(y==0 & yh==0)/sum(y==0); 
acc= sum(y==yh)/length(y); 
end