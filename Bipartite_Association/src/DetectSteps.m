function [stepDetector,walkingFrequency] = DetectSteps(time,accel)
% Signal Preprocessing: Uncomment the next 2 lines if accel signal is raw
% accel = smooth(accel,21, 'sgolay',1);
% accel = detrend(accel,5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepDetector = zeros(size(accel,1),1);
stepLength = zeros(size(accel,1),1);
velocity = zeros(size(accel,1),1);
i=1;
segNo = 0;
window = 50;
walkingFrequency = zeros(size(accel,1),1);
wf = [];
stepCount = 0;
% while window<=size(accel,1)
    minPeakHeight = std(accel);
%     segmentTime = time(i:window);
    [pks,locs] = findpeaks(accel, 'MinPeakProminence',3);
% [pks,locs] = findpeaks(accel(1:window),'MinPeakHeight',0.8)
%     for h=1:size(locs)
%     locs = locs(diff(locs)<=20);
    stepDetector(locs) = 1;
%     stepTimes = time(locs);
    
%     end
%     stepDetector(i:window) = mean(accelmag(i:window))>=minPeakHeight;
%     if ~isempty(pks)
%         stepAccel = pks;
%         stepTime = time(locs);
% %         stepCount(locs) = length(pks);
%         stepInterval = seconds(diff(time(locs))); 
%         stepInterval = stepInterval(stepInterval<=1);
% %         walkingFrequency(locs) = stepCount./seconds(time(end)-time(1));%step/sec
%         InstepAcc = pks(stepInterval<=1);
%     end 
%     for b=1:length(stepInterval)
%         stepLengths(b) = trapz(stepInterval(b),InstepAcc(b),1);
%     end
%     if isempty(walkingFrequency)
%         wf = [wf;0];
%     else
%         wf = [wf;walkingFrequency];
%     end
%     WF = mean(wf);
%     stepLength = WF
%     i = window+1;
%     window = window+50;
end

            