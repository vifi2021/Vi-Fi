function [Dist] = CalcDistance(timestamp,Fmed)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    Dist = [sqrt(diff(Fmed(:,1)).^2+...
                            diff(Fmed(:,2)).^2);0];
                        
end

