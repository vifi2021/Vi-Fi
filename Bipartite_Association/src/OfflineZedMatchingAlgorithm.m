len = max(max(cellfun('size',Ad,1)));
MissingAd = NaN(len,1);
AllWeights = cell(size(Ad,1),size(Ad,2));
for r=1:size(AllWeights,1)
    for c=1:size(AllWeights,2)
        AllWeights{r,c} = MissingAd;
        AllWeights{r,c}(1:size(Ad{r,c},1))=Ad{r,c};
    end
end
Ad = AllWeights;
TP = 0;
FP = 0;
TN = 0;
FN = 0;

TPs=0;
FPs=0;
Accuracy = [];
HeadStd = [];
mulTimeSegment = 1;
% subTP = nan(1,size(IsHere,2)-1);
% subTP(gndtrackId) = 0;
CorrelationResult = zeros(size(IsHere,1),size(IsHere,2));
correlated = zeros(1,size(IsHere,2)-1);
distM = cellfun(@(v)v(1),Ad);
% MinDist = inf(size(Ad,1),size(Ad,2));
for f = 1 : size(IsHere,1)
    assrow=[];
    [~,currentTrackids] = find(IsHere(f,2:end));
    FrameMinDist = distM.*IsHere(f,2:end);
    FrameMinDist(FrameMinDist==0|isnan(FrameMinDist))=inf;
    FrameMinDistGraph = FrameMinDist;
    dim = size(FrameMinDistGraph);
    [max_nodes,max_dim] = max(dim);
    if max_dim == 2
        num_of_additional_phones = max_nodes - size(FrameMinDistGraph,1);
        Graph_FrameMinDist = [FrameMinDistGraph;inf(num_of_additional_phones,size(FrameMinDistGraph,2))];
    else % max_dim is 1
        num_of_additional_subjects = max_nodes - size(FrameMinDistGraph,2);
        Graph_FrameMinDist = [FrameMinDistGraph,inf(size(FrameMinDistGraph,1),num_of_additional_subjects)];
    end
    [assignment,cost] = munkres(Graph_FrameMinDist);
    [assrow,asscol] = find(assignment);

    CorrelationResult(f,1) = IsHere(f,1);
    CorrelationResult(f,asscol+1) = assrow'.*IsHere(f,asscol+1);
 
    mulTimeSegment = mulTimeSegment+global_winShift;
    
    
end
TotalPeople = 0;
matchID = 0;
cd(sequences_path)
load(subFolders(k).name+"NoOfPohneHolsers")

if day == "20211004"
    AllphonesHolders = {'Subject24', 'Subject19'};
    subjects = {'Subject24', 'Subject19'};
else
    AllphonesHolders = {'Subject24', 'Subject19', 'Subject11'};
    subjects = {'Subject24', 'Subject19', 'Subject11'};
end

for ai = 1:size(CorrelationResult,1)
    phoneTP = [];
    trackIDTP = [];
    phoneFP = [];
    trackIDFP = [];
    phoneGND = [];
    fTP = 0;
    fFP = 0;
    fFN = 0;
    fTN = 0;
    [~,trackID]  = find(IsHere(ai,2:end)>0);
    peopleInFrame = sum(IsHere(ai,2:end)>0);
    TotalPeople = TotalPeople+peopleInFrame;
    frno = CorrelationResult(ai,1);
    for ti=1:length(trackID)
        TrackIDSub = string(ZedImgPosition{1,trackID(ti)}.SubName(1));
        if (CorrelationResult(ai,trackID(ti)+1)==0) && (IsHere(ai,trackID(ti)+1)==1)
            if ~any(find(TrackIDSub==phonesHolders))
                ZedImgPosition{1,trackID(ti)}.AssignedPhone(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = "None";
                ZedImgPosition{1,trackID(ti)}.AssPhoneID(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = -1;
                TN = TN+1;
                fTN = fTN+1;
            else
                ZedImgPosition{1,trackID(ti)}.AssignedPhone(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = "None";
                ZedImgPosition{1,trackID(ti)}.AssPhoneID(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = -1;
                FN = FN+1;
                fFN = fFN+1;
            end
        else
            phone = phonesHolders{CorrelationResult(ai,trackID(ti)+1)};
            phone = convertCharsToStrings(phone);
            
            ZedImgPosition{1,trackID(ti)}.AssignedPhone(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = phone;
            ZedImgPosition{1,trackID(ti)}.AssPhoneID(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = CorrelationResult(ai,trackID(ti)+1);
            %             ZedImgPosition{1,trackID(ti)}.Distance(ZedImgPosition{1,trackID(ti)}.frameNo==frno) = confidenceMatrix(ai,trackID(ti)+1);
            
            %               TracktorgndPosition{1,trackID(ti)}.AssignedPhone = repmat(phone,size(TracktorgndPosition{1,trackID(ti)},1),1);
            phoneGND = [phoneGND,find(phonesHolders==TrackIDSub)];
            ModDecision{end+1,1} = k;
            ModDecision{end,2} = CorrelationResult(ai,1);
            if strcmp(TrackIDSub,phone)
                %                 subjTP(trackID(ti)) = subjTP(trackID(ti))+1;
                TP = TP+1;
                fTP = fTP+1;
                sTP(CorrelationResult(ai,trackID(ti)+1))=sTP(CorrelationResult(ai,trackID(ti)+1))+1;
                phoneTP = [phoneTP,CorrelationResult(ai,trackID(ti)+1)];
                trackIDTP = [trackIDTP,trackID(ti)];
            else
                FP = FP+1;
                sFP(CorrelationResult(ai,trackID(ti)+1))=sFP(CorrelationResult(ai,trackID(ti)+1))+1;
                phoneFP = [phoneFP,CorrelationResult(ai,trackID(ti)+1)];
                
                trackIDFP = [trackIDFP,trackID(ti)];
            end
        end
    end
    %             TN = TN + peopleInFrame-(FP+TP+FN)
    frameAcc(ai) = fTP/peopleInFrame;
    frameTP(ai) = fTP;
    %detected_people(ai)=peopleInFrame;
    cd(sequences_path)
    detected_people(ai) = NoOfPohneHolsers(ai);
    frame_number(ai) = ai;
    seq_num(ai) = k;
    Frames_FrameMinDist{ai,1}=frameAcc(ai);
    x1 = cellfun(@(v)v(1),Adindiv(:,trackID));
    x2 = cellfun(@(v)v(2),Adindiv(:,trackID));
    if size(Adindiv(:,trackID),2)>2
        x3 = cellfun(@(v)v(3),Adindiv(:,trackID));
    else
        x3 = [];
    end
    Frames_FrameMinDist{ai,2} = x1;
    Frames_FrameMinDist{ai,3} = x2;
    Frames_FrameMinDist{ai,4} = x3;
    Frames_FrameMinDist{ai,5}=detected_people(ai);
    Frames_FrameMinDist{ai,6}=phoneTP;
    Frames_FrameMinDist{ai,7}=phoneFP;
    Frames_FrameMinDist{ai,8}=phoneGND;
    Frames_FrameMinDist{ai,9}=trackIDTP;
    Frames_FrameMinDist{ai,10}=trackIDFP;
end
total_frameAcc = [total_frameAcc;seq_num',frame_number',frameAcc',detected_people',frameTP'];
%%
overallFrameAccuracy = nansum(frameAcc(1,:))/sum(~isnan(frameAcc(1,:)));

ACC = TP/sum(detected_people)

%ACC = TP/sum(detected_people)
PR = TP / (TP + FP);
REC = TP / (TP + FN);
NPV = TN / (TN + FP +FN + TP);
FS = 2 * (PR * REC) / (PR + REC);

AllSeqAcc = [AllSeqAcc,ACC]
cd(sequences_path+"/"+subFolders(k).name)
fname = subFolders(k).name+"CorrelationResultOffline.mat";
save(fname,'CorrelationResult');
%%
