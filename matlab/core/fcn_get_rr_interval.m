function [interval,itimes,rr,R_loc] = get_rr_interval(epoch,fs )
[qrs_amp_raw,R_loc,delay]=fcn_pan_tompkin(epoch,fs,0);
itimes = R_loc/fs;
interval=diff(itimes);
nintervals = length(interval);
itimes = itimes(1:nintervals);
rr=itimes*fs;
amp=epoch(R_loc);
end

