close all;
clear;
%DataPreparation

frameCounter = 1;
frameLength = [];
AllSequencesCorrelationData = {};
ALLcount = 1;
AllcountImg = 1;
fff = 1;

% samplingRate = 10;
samplingRate = 3;
specificDay = 0;
testSequences = 1;
% testSequences = 1;

%trial = 1;
for trial = 5:5
    cd('D:\RANProject\src2\src')
    if ismember(trial,[1,2])
        continue
    end
    tic
    total_frameAcc = [];
    BayesSeqAccuracy = [];
    trial
    num_of_phones = 3;
    AllTPs = 0;
    AllFPs = 0;
    AllTNs = 0;
    AllFNs = 0;
    AllSeqFS = [];
    AllSeqRecall =  [];
    AllSeqPres = [];
    AllSeqNPV = [];
    totalbb=0;
    SeqAcc = 0;
    AllSeqAcc = [];
    ModDecision = {};
    TrackIdLength = [];
    MatchedFrameDistSTD = [];
    CorrelationVariation = {};

    ZedSequencesCorrelationOnline;
%     ZedSequencesCorrelationOffline;
    %     calculateZedAccuracy2021
    %All_sequences_total_frameAcc = [All_sequences_total_frameAcc;total_frameAcc];
    All_ModalityDecision{trial} = ModDecision;
    AvgSeqAcc = SeqAcc/5;
    AllSeqAcc
    AllModalityAccuracy(trial,:) = AllSeqAcc;
    ROCmat(trial,:) = [AllTPs,AllFPs,AllTNs,AllFNs,AllTPs+AllFPs+AllTNs+AllFNs,totalbb,mean(AllSeqPres),mean(AllSeqRecall),mean(AllSeqFS),mean(AllSeqAcc)];
    AccuracyMatrix(trial,:) = [AvgSeqAcc,mean(AllSeqFS),mean(AllSeqRecall),mean(AllSeqPres)];
    BoxPlotMat{trial} = [AllSeqPres',AllSeqRecall',AllSeqNPV',AllSeqFS',AllSeqAcc'];
    %trial = trial+1;
    toc
    % save("Frames_FrameMinDistTrial"+num2str(trial),'Frames_FrameMinDist')
    cd('D:\RANProject\src2\src')
end
