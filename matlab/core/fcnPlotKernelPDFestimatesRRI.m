function [f0,f1]= fcnPlotKernelPDFestimatesRRI(x,y0,y1);

% kernel density estimation
% f0 = ksdensity(y0,x,'function','pdf','bandwidth',0.25);
% f1 = ksdensity(y1,x,'function','pdf','bandwidth',0.25);

% plots
f0=hist(log(y0),x); f0=f0/sum(f0);
f1=hist(log(y1),x); f1=f1/sum(f1);

plot(x,log(f0),'b');
hold on
plot(x,log(f1),'r');

hh0 = createPatches(x,log(f0),.01,'b',.2);
hold on
hh1 = createPatches(x,log(f1),.01,'r',.2);

end