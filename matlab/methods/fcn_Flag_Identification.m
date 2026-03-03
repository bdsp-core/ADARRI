function [yy]=fcn_Flag_Identification(y,Y,theta)

yy=zeros(size(Y));
for n=1:length(y)-2
    if Y(1,n)>theta; % above threshold = FLAG ON
        yy(1,n)=1;
        if n<1 
            if Y(n-1)<theta
                if y(n)-y(n+1)<0
                    exit=2; % long beat
                else exit=1; % short beat
                end
            else exit=0; yy(1,n)=1; % can't evaluate, FLAG ON -- END
            end
            if exit==1 % Short beat
                if Y(n+2)<theta
                    if y(n-1)<y(n+1)
                        x=y(n-1)+y(n);
                    else x=y(n+1)+y(n);
                    end
                    if x-y(n-1)>theta && x-y(n+1)>theta
                        yy(1,n)=0; % FLAG OFF -- END
                    else yy(1,n)=1; % FLAG ON -- END
                    end
                else yy(1,n)=1; % can't evaluate, FLAG ON -- END
                end
            elseif exit==2 % long beat
                if Y(n+2)<theta
                    x=y(n)/2;
                    if x-y(n-1)<-theta && x-y(n+1)<-theta
                        yy(1,n)=0; % FLAG OFF -- END
                    else yy(1,n)=1; % FLAG ON -- END
                    end
                else yy(1,n)=1; % can't evaluate, FLAG ON -- END
                end
            end
        end
    end
end