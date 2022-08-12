clear
close all
% samplingRate = 10;
samplingRate = 3;
specificDay = 0;
%testSequences = 0;
testSequences = 1;
missing =1;

if testSequences == 1
    sequences_path = 'E:/RANProject/TestSequences/';% change this path to where you are storing the sequences folders
end
% if testSequences == 0
%     sequences_path = 'E:/RANProject/WINLABData/';% change this path to where you are storing the sequences folders - make it as the above 
% end
% if missing == 1
%     sequences_path = 'E:/RANProject/WINLABDataset/RAN_IMU_only_MISSING/RAN_IMU_only_MISSING';% change this path to where you are storing the sequences folders - make it as the above 
% end
% for scen=1:1
%     sequences_path = "D:\RANProject\WINLABDataset\Scenes\scene"+num2str(scen-1);% change this path to where you are storing the sequences folders
%     
   
    disp(samplingRate)
    src_path = "E:\RANProject\src2\src"; % change this path to where you are keeping the scripts

    cd(src_path)
    files = dir(sequences_path+"/*");
    files(ismember( {files.name}, {'.', '..'})) = [];  %remove . and ..
    % Get a logical vector that tells which is a directory.
    dirFlags = [files.isdir];
    % Extract only those that are directories.
    subFolders = files(dirFlags);



    for k = 1 : size(subFolders,1)%for each trial
        k
    %     if k~=[1,2,5,7,8,9,10]
    %         continue
    %     end
        day = subFolders(k).name(1:8);
        fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
        IMU
        cd(src_path)
        EventDetector
        cd(src_path)
        ZedIMUDeadReckoninglive
%         cd(src_path)
%         getFTM
%         cd(src_path)
%         Zed2GNDMatching
%         cd(src_path)
%         ZedPositionlive2
%         cd(src_path)
    end
% end