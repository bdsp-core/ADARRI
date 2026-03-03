function [Rloc]=fcn_calculate_absdiff(adRRI,theta1)

% abs(diff(RRI))
Rloc=zeros(1,length(adRRI));

for n=1:length(adRRI)
    if adRRI(n)>theta1
        Rloc(1,n)=1;
    end
end

% Als er opeenvolgend <1 rood zijn, wordt de laatste groen
for l=3:length(Rloc)
    if Rloc(l-2)==1 && Rloc(l-1)==1 && Rloc(l)==0
    else Rloc(l-1)=0; 
    end
end

