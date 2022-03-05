ClearExeptions
% close all
% % Get a list of all files and folders in this folder.
% files = dir("C:/RANProject/../*");
% files(ismember( {files.name}, {'.', '..'})) = [];  %remove . and ..
% % Get a logical vector that tells which is a directory.
% dirFlags = [files.isdir];
% % Extract only those that are directories.
% subFolders = files(dirFlags);
% % Print folder names to command window.
% figure
% for k = 1 : size(subFolders,1)%for each trial
%     k
    
%     fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
    disp(sequences_path+"/"+subFolders(k).name+"/IMU/*.csv")
    seqname = subFolders(k).name;
    [~, PhoneIMUReadings] = ImportPhoneIMU(subFolders(k).name,dir(sequences_path+"/"+subFolders(k).name+"/IMU/*.csv"),src_path,sequences_path,seqname);

    fname = subFolders(k).name+"IMU";
    cd(sequences_path+"/"+subFolders(k).name)
    save(fname,'PhoneIMUReadings')
    cd(src_path)
% end
%%