% samplingRate = 10;
samplingRate = 3;
specificDay = 0;
%testSequences = 0;
testSequences = 1;

if testSequences == 1
    sequences_path = 'D:/RANProject/TestSequences/';
end
if testSequences == 0
    sequences_path = 'D:/RANProject/WINLABData/';
end

% Get a list of all files and folders in this folder.

disp(samplingRate)
src_path = "D:\RANProject\src2\src";

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
    cd(src_path)
    getFTM
    cd(src_path)
    Zed2GNDMatching
    cd(src_path)
    ZedPositionlive2
    cd(src_path)
end