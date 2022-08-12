function [yaw,pitch,roll] = getPhoneOrientation(accel,mag,gyro)

accel(:,1) = 0;
accel(:,2) = 0;

SampleRate = 50;

idxstart = 1;
idxend = length(accel(:,1));

% %https://www.mathworks.com/help/nav/ug/estimate-phone-orientation-using-sensor-fusion.html
a = [accel(idxstart:idxend,1), accel(idxstart:idxend,2), accel(idxstart:idxend,3)];
w = [gyro(idxstart:idxend,1), gyro(idxstart:idxend,2), gyro(idxstart:idxend,3)];
m = [mag(idxstart:idxend,1), mag(idxstart:idxend,2), mag(idxstart:idxend,3)];


aFilter = ahrsfilter('SampleRate',SampleRate,'ReferenceFrame','NED'); 

reset(aFilter);

q = aFilter(a,w,m);

orientation(:,1) = aFilter(a,w,m); % Note : Yaw is calculated with respect to the magnetic north
eulFilt = eulerd(q,'ZYX','frame');

yaw = eulFilt(:,1);
roll = eulFilt(:,3);
pitch = eulFilt(:,2);

release(aFilter)

end

