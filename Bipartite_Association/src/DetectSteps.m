function stepDetector = DetectSteps(time,accel)

stepDetector = zeros(size(accel,1),1);
minPeakHeight = std(accel);
[pks,locs] = findpeaks(accel, 'MinPeakProminence',3);
stepDetector(locs) = 1;

end

            