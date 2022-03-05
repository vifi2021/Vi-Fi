% ImgPosition{1, sub} = retime(ImgPosition{1, sub},'regular','nearest','TimeStep',dt);
[C,ia,ic] = unique(ZedImgPosition{1,sub}.timestamp);
ZedImgPosition{1, sub} = ZedImgPosition{1, sub}(ia,:);
% S = withtol(ImgPosition{1, sub}.timestamp,milliseconds(10));
sysTime = ZedframeTimestamp;
tlower = datetime(ValidTimestamps.VarName3(ValidTimestamps.VarName1==subFolders(k).name), 'Format', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
tupper = datetime(ValidTimestamps.VarName5(ValidTimestamps.VarName1==subFolders(k).name), 'Format', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
tf = isbetween(sysTime,tlower,tupper);
ValidFrameNo = zedFrameNo(tf);
sysTime = sysTime(tf);
sysTime.TimeZone = 'America/New_York';
% sysTime.Day = 23;
% sysTime.Year = 2020;
% sysTime.Month = 12;
% ShowingframeTimestamp = sysTime(S,:);
ZedImgPosition{1, sub}.timestamp.TimeZone = 'America/New_York';
IIsHere(:,1) = ValidFrameNo;
IsHere(:,sub+1) = ismember(sysTime,ZedImgPosition{1, sub}.timestamp);% ImgPosition{1, sub} = ImgPosition{1, sub}(S,:);
IMU = imuReadings{1,phone};
IMU.timestamp.TimeZone = 'America/New_York';
% IMU.timestamp.Day = 23;
% IMU.timestamp.Year = 2020;
% IMU.timestamp.Month = 12;
IMU.PhoneID = [];
VID = ZedImgPosition{1,sub};
VID.timestamp.TimeZone = 'America/New_York';
% VID.timestamp.Day = 23;
% VID.timestamp.Year = 2020;
% VID.timestamp.Month = 12;
% VID.subject = [];
VID.SubName = [];
% imuvisual = synchronize(VID,IMU,'first','nearest');
%
phoneFTM{1,phone} = AllFTMData{1,phone};
FTMTable = phoneFTM{1,phone};
FTMTable.FName = [];
%

% visual = ImgPosition{1, sub}(isbetween(ImgPosition{1, sub}.timestamp,lowert,uppert),:);
% imu = PhoneIMUReadings{1,phone}(isbetween(PhoneIMUReadings{1,phone}.timestamp,lowert,uppert),:);

% MFTMTable = FTMCorrelatorData(FTMCorrelatorData.trackID==sub-1,[1,2+phone]);
% MFTMTable = table2timetable(MFTMTable);
% MFTMTable.timestamp.TimeZone = 'America/New_York';

%imuvisFTM{phone,sub} = synchronize(VID,IMU,FTMTable,'first','nearest');
imuvisFTM{phone,sub} = synchronize(VID,IMU,FTMTable,'first','nearest');

phonesHolders = {'Subject24','Subject19','Subject11','None'};

overlap = global_overlap;
windSize = global_windSize;
winShift = global_winShift;
Checkpoint = 1;
d1 = [];
headCorr = [];
trajdtw = [];
ftmDist = [];
Mftmdist = [];
dtwvisimu = [];
cuurImuVis = [];
trans = [];
cT = [];
VisCoor = [];
i=1;

Localframe = 1;
headWeight = global_headWeight;
trajWeight = global_trajWeight;
FTMWeight = global_FTMWeight;
mftmWeight = global_MFTMWeight;
% visual.subject(2);
% imu.PhoneID(1)
% minSize = min([size(visual,1) size(imu,1)])
%while  windSize <= size(sysTime,1)
windSize = size(sysTime,1);
frame = frame+1;
nimuTraj =[NaN,NaN];
nvidTraj =[NaN,NaN];
nvidheading =NaN;
nimuyaw =NaN;
Z =[NaN,NaN];

%    fi = zedFrameNo(i);

%   fwind = zedFrameNo(windSize);
if sum(IsHere(1:windSize,sub+1))>0
    cuurImuVis = imuvisFTM{phone,sub}(isbetween(imuvisFTM{phone,sub}.timestamp,sysTime(1),sysTime(windSize)),:);
    AllPhonesTrackIDs{phone,sub} =  cuurImuVis;
    %         Mftmdist(Localframe) = MFTMTable{:,1}*mftmWeight;
    %         mftm = timetable2table(mftm);
    %         mftm = table2array(mftm(end,2));
    % mftm = mean(mftm);
    %Mftmdist(Localframe) = mftm;%*mftmWeight;
    %
    cuurImuVis.depth(isnan(cuurImuVis.depth)|isinf(cuurImuVis.depth))=0;
    
    
    %         [maxVar,maxInd] = max([cuurImuVis.hvar(end),cuurImuVis.fvar(end)]);
    %         if maxInd==1
    %             FTMWeight = 0.1;
    %             headWeight = 0.8;
    %         else
    %             FTMWeight = 0.8;
    %             headWeight = 0.1;
    %         end
    
    ftmDist(Localframe) = dtw(cuurImuVis.depth,cuurImuVis.FTM)/length(cuurImuVis.depth);
    
    
    nimuTraj = [cuurImuVis.xPos cuurImuVis.yPos];
    
    nvidTraj = cuurImuVis.Fmed;
    
    [~,Z,tr] = procrustes(nvidTraj,nimuTraj);
    
    %         dx = [diff(Z(:,1));0];
    %         dy = [diff(Z(:,2));0];
    %         ZHeadingx = rad2deg(atan2(dy,dx));
    %         ZHeadingy = rad2deg(atan2(dx,dy));
    %dtw(nvidTraj(:,1),Z(:,1))
    %dtw(nvidTraj(:,2),Z(:,2))
    d = (dtw(nvidTraj(:,1),Z(:,1))/length(nvidTraj(:,1)))+(dtw(nvidTraj(:,2),Z(:,2))/length(nvidTraj(:,1)));
    %         d = dtw(nvidTraj(:,1),Z(:,1))+dtw(nvidTraj(:,2),Z(:,2));
    d1(Localframe) = d;%*trajWeight;
    
    %        if sub==54
    %             subplot(5,1,phone)
    %             plot(nvidTraj,Z,'LineWidth',2)
    %             legend({'Visual Trajectory','Transformed Inertial Trajectory'})
    %             title("Distance = "+num2str(d1(Localframe)))
    %        end
    
    ntrajdtw = dtw(nvidTraj,nimuTraj);
    trajdtw(Localframe) = ntrajdtw;
    
    % standerdize heading data from both sources before using DTW
    nimuyaw = cuurImuVis.yaw;
    nimuyaw(nimuyaw<0) = nimuyaw(nimuyaw<0)+360;
    %                 if std(nimuyaw)~=0
    %                     nimuyaw = normalize(nimuyaw,'scale');
    %                 end
    %
    nvidheading = cuurImuVis.visHeadingy;
    nvidheading(nvidheading<0) = nvidheading(nvidheading<0)+360;
    
    cdt = dtw(nimuyaw,nvidheading)/length(nimuyaw);
    %         cdt = dtw(nimuyaw,nvidheading);
    %         zcdt = dtw(ZHeadingy,nvidheading);
    headCorr(Localframe) = cdt;%*headWeight;
    
    %         if sub==54
    %            % dtw(nimuyaw,nvidheading)
    %             if phone==1
    %                 figure
    %             end
    %             subplot(5,3,phone)
    %             plot(nvidheading,'LineWidth',2)
    %             hold
    %             plot(nimuyaw,'LineWidth',2)
    %             legend({'Visual Heading','Phone Heading'})
    %             title("Distance = "+num2str(headCorr(Localframe)))
    %        end
    
    
    
    %         AllSequencesCorrelationData(frameCounter,:) = {k,phone,sub,IsHere(i,1),datestr(sysTime(windSize)),headCorr(Localframe),d1(Localframe),ftmDist(Localframe),...
    %             cuurImuVis.hvar(end),cuurImuVis.pophvar(end),...
    %             cuurImuVis.fvar(end),cuurImuVis.popfvar(end)};
    %         AllSequencesCorrelationData(frameCounter,:) = {k,phone,sub,IsHere(i,1),datestr(sysTime(windSize)),headCorr(Localframe),d1(Localframe),ftmDist(Localframe),Mftmdist(Localframe)};
    %        AllSequencesCorrelationData(frameCounter,:) = {k,phone,sub-1,IsHere(i,1),datestr(sysTime(windSize),'yyyy-mm-dd HH:MM:SS.FFF'),d1(Localframe),headCorr(Localframe),Mftmdist(Localframe)};
    %          figure('name',subFolders(k).name)
    %             subplot(1,2,1)
    %             plot(PhoneIMUReadings{1,j}.xPos,PhoneIMUReadings{1,j}.yPos,'b','LineWidth',2);
    %                 %seconds(PhoneIMUReadings{1,j}.timestamp-PhoneIMUReadings{1,j}.timestamp(1)),'LineWidth',2);
    %             xlabel('X [m]');
    %             ylabel('Y [m]');
    % %             zlabel('time');
    %             title (PhoneIMUReadings{1,j}.PhoneID(2) + " IMU Trajectory");
    %             grid on
    %             hold on
    %
    %             subplot(1,2,2)
    %             plot(vid.Fmed(2:end,1),vid.Fmed(2:end,2),'LineWidth',2);%seconds(vid.timestamp(2:end)-vid.timestamp(2)),'LineWidth',2)
    %             hold on
    %             [~,Z,tr] = procrustes([vid.Fmed(2:end,1),vid.Fmed(2:end,2)],[PhoneIMUReadings{1,j}.xPos,PhoneIMUReadings{1,j}.yPos]);
    %             plot(Z(:,1),Z(:,2),'LineWidth',2)
    %             xlabel('X [pixel]'); ylabel('Y [pixel]');
    %             %zlabel('time');
    %             title(vid.subject(2) + " ")
    %             legend({'Vid. Trajectory','Transformed Phone Traj.'})
    %             grid on
    %             hold off
    
else
    headCorr(Localframe) = NaN;
    trajdtw(Localframe) = NaN;
    d1(Localframe) = NaN;
    ftmDist(Localframe) = NaN;
    Mftmdist(Localframe) = NaN;
end

i = i + winShift;
windSize = windSize + winShift ;
if windSize >= size(sysTime,1)
    windSize = size(sysTime,1);
end
frameCounter = frameCounter+1;
Localframe = Localframe+1;

%end

d1 = d1';
headCorr = headCorr';
trajdtw = trajdtw';
ftmDist = ftmDist';
% if size(Mftmdist,2)>1
%     Mftmdist = Mftmdist';
% end
% headCorr = normalize(headCorr,'range')
% trajdtw = normalize(trajdtw,'range')
% ftmDist = normalize(ftmDist,'range')
% if std(headCorr)~=0
%     headCorr = normalize(headCorr);
% end
%
% if std(trajdtw)~=0
%     trajdtw = normalize(trajdtw);
% end
%
% if std(ftmDist)~=0
%     ftmDist = normalize(ftmDist);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AdNoWeights{phone,sub} = [d1,headCorr];
% if (isnan(headCorr) && ~isnan(d1)) || (headCorr == 0 && d1 ~=0)
%     headWeight = 0;
%     trajWeight = 1;
% else

% end
%Adindiv{phone,sub} = [d1 , headCorr, MFTMTable{:,1}];
Adindiv{phone,sub} = [d1 , headCorr, ftmDist];
%Ad{phone,sub} = headCorr +  d1 + MFTMTable{:,1}*mftmWeight;% similarity of all phones with all subjects
%Ad{phone,sub} = headCorr +  d1 + MFTMTable{:,1}.*mftmWeight;
Ad{phone,sub} = headCorr.*headWeight +  d1.*trajdtWeight + ftmDist*FTMWeight;
% Ad{phone,sub} = headCorr.*headWeight ;
alltrans{phone,sub} = trans;
%         norm_dtwvisimu = (dtwvisimu - min(dtwvisimu)) / ( max(dtwvisimu) - min(dtwvisimu) );
alld{sub} = d1(~isempty(d1)).*trajWeight + headCorr(~isempty(headCorr)).*headWeight + trajdtw(~isempty(trajdtw)).*trajdtWeight; % dissimilarity of all subjects with current phone
% alld{sub} = headCorr.*headWeight;
% allcT{phone,sub} = cT;