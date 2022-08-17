function [yaw,pitch,roll] = getPhoneOrientation(accel,mag)
% x:pitch , y:roll , z:yaw

% MATLAB compass function:
q = ecompass(accel,mag);
orientation = eulerd(q,'ZYX','frame');
yaw = orientation(:,1);
roll = orientation(:,2);
pitch = orientation(:,3);
end


