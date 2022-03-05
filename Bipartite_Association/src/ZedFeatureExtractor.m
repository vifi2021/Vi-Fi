
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
IsHere(:,1) = ValidFrameNo;
IsHere(:,sub+1) = ismember(sysTime,ZedImgPosition{1, sub}.timestamp);
% ImgPosition{1, sub} = ImgPosition{1, sub}(S,:);
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

% MFTMTable = FTMCorrelatorData(FTMCorrelatorData.trackID==sub-1,[1,3+phone]);
% MFTMTable = table2timetable(MFTMTable);
% MFTMTable.timestamp.TimeZone = 'America/New_York';

%imuvisFTM{phone,sub} = synchronize(VID,IMU,FTMTable,MFTMTable,'first','nearest');
imuvisFTM{phone,sub} = synchronize(VID,IMU,FTMTable,'first','nearest');

phonesHolders = {'Subject24', 'Subject19', 'Subject11','None'};
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

while  i <= size(sysTime,1)
    %     windSize
    frame = frame+1;
    nimuTraj =[NaN,NaN];
    nvidTraj =[NaN,NaN];
    nvidheading =NaN;
    nimuyaw =NaN;
    Z =[NaN,NaN];
    
    fi = zedFrameNo(i);
    
    fwind = zedFrameNo(windSize);
    if sum(IsHere(windSize,sub+1))>0
        cuurImuVis = imuvisFTM{phone,sub}(isbetween(imuvisFTM{phone,sub}.timestamp,sysTime(1),sysTime(windSize)),:);
        
       
        cuurImuVis.depth(isnan(cuurImuVis.depth)|isinf(cuurImuVis.depth))=0;
        ftmDist(Localframe) = dtw(cuurImuVis.depth,cuurImuVis.FTM)/length(cuurImuVis.depth);
        
        
        nimuTraj = [cuurImuVis.xPos cuurImuVis.yPos];
        
        nvidTraj = cuurImuVis.Fmed;
        
        [~,Z,tr] = procrustes(nvidTraj,nimuTraj);
        cuurImuVis.TransformedxPos = Z(:,1);
        cuurImuVis.TransformedyPos = Z(:,2);
     
        
        d = (dtw(nvidTraj(:,1),Z(:,1))/length(cuurImuVis.depth))+(dtw(nvidTraj(:,2),Z(:,2))/length(cuurImuVis.depth));
        
        d1(Localframe) = d;%*trajWeight;
        
        ntrajdtw = dtw(nvidTraj,nimuTraj);
        trajdtw(Localframe) = ntrajdtw;
        
        % standerdize heading data from both sources before using DTW
        nimuyaw = cuurImuVis.yaw;
        nimuyaw(nimuyaw<0) = nimuyaw(nimuyaw<0)+360;
        cuurImuVis.nimuyaw = nimuyaw;
     
        nvidheading = cuurImuVis.visHeadingy;
        nvidheading(nvidheading<0) = nvidheading(nvidheading<0)+360;
        cuurImuVis.nvidheading = nvidheading;% to save it after modified
        
        cdt = dtw(nimuyaw,nvidheading)/length(cuurImuVis.depth);
       
        headCorr(Localframe) = cdt;%*headWeight;
        
     
       
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
    
end
if  ~isempty(cuurImuVis)
           %idx = ismember(cuurImuVis.frameNo,3727:3799);
           %if sum(idx)>0%
           frameInfo{phone,sub} = cuurImuVis;%{phone,sub,cuurImuVis.frameNo(idx),cuurImuVis.timestamp(idx),cuurImuVis.depth(idx),...
           fff=fff+1;
           %end
end

d1 = d1';
headCorr = headCorr';
trajdtw = trajdtw';
ftmDist = ftmDist';


Adindiv{phone,sub} = [d1, headCorr, ftmDist, headCorr.*headWeight +  d1.*trajWeight, headCorr.*headWeight +  d1.*trajWeight + Mftmdist.*mftmWeight];
%Ad{phone,sub} = headCorr.*headWeight +  d1.*trajWeight + Mftmdist.*mftmWeight;% + ;% similarity of all phones with all subjects
switch trial
    case 1
        Ad{phone,sub} = d1.*trajWeight;
    case 2
        Ad{phone,sub} = headCorr.*headWeight;
    case 3
        Ad{phone,sub} = headCorr.*headWeight +  d1.*trajWeight;
    case 4
        Ad{phone,sub} = ftmDist.*FTMWeight;
    case 5
        %Ad{phone,sub} = (headCorr +  d1 + ftmDist)/3;
        Ad{phone,sub} = headCorr.*headWeight +  d1.*trajWeight + ftmDist.*FTMWeight;
        %Ad{phone,sub} = mean([headCorr,d1,ftmDist],2);
end
% Ad{phone,sub} = headCorr.*headWeight ;
alltrans{phone,sub} = trans;

%         norm_dtwvisimu = (dtwvisimu - min(dtwvisimu)) / ( max(dtwvisimu) - min(dtwvisimu) );
alld{sub} = d1(~isempty(d1)).*trajWeight + headCorr(~isempty(headCorr)).*headWeight + trajdtw(~isempty(trajdtw)).*trajdtWeight; % dissimilarity of all subjects with current phone
