ClearExeptions
% close all
% cd('C:/RANProject')
% files = dir("C:/RANProject/../*");
% files(ismember( {files.name}, {'.', '..'})) = [];  %remove . and ..
% % Get a logical vector that tells which is a directory.
% dirFlags = [files.isdir];
% % Extract only those that are directories.
% subFolders = files(dirFlags);
% % Print folder names to command window.
%%
% 
% for k = 1 : size(subFolders,1)
%     
    %IMPORT FTM DATA
    FTMfiles = dir(sequences_path+"/"+subFolders(k).name+"/WiFi/*.csv");
    num_depthfiles = length(FTMfiles);
    subplot(3,5,k)
    for kf = 1:num_depthfiles
        FTMData = importFTMfile(sequences_path+"/"+subFolders(k).name+"/WiFi/"+FTMfiles(kf).name);
        FTMData.timestamp = FTMData.timestamp/10^3;
        FTMData.timestamp = datetime(FTMData.timestamp,'ConvertFrom','posixtime','timezone','America/New_York','Format','HH:mm:ss.SSSSSS');
        FTMData = table2timetable(FTMData);
        FTMData.FTM = FTMData.FTM./1000;
        FTMData.FName = repmat(FTMfiles(kf).name,size(FTMData,1),1);
        AllFTMData{1,kf}=FTMData;
        plot(AllFTMData{1,kf}.timestamp,AllFTMData{1,kf}.FTM+kf*2)
        title(subFolders(k).name)
        phname{kf} = AllFTMData{1,kf}.FName(1);
        ylabel('FTM [m]')
        hold on
    end
    legend(phname)
    hold off
    
%%

    fname = subFolders(k).name+"FTM";
    cd(sequences_path+"/"+subFolders(k).name)
    save(fname,'AllFTMData')
    cd(sequences_path)
% end