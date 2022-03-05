ClearExeptions
close all
% sequences_path = "C:/RANProject/..";
% src_path = "C:/RANProject/src";

AllIMU = {};
AllPositions = {};
PhoneIMUReadings = {};
fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
cd(sequences_path+"/"+subFolders(k).name)


IMUfname = subFolders(k).name+"IMU.mat";
AllIMU = load(IMUfname);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f1 = figure('name',subFolders(k).name)
alltable = [];
for i=1:size(AllIMU.PhoneIMUReadings,2)
    cd(src_path)
 
    [stepDetector,walkingFrequency] = DetectSteps(AllIMU.PhoneIMUReadings{1,i}.timestamp,AllIMU.PhoneIMUReadings{1,i}.gm);
    
    cd(sequences_path+"/"+subFolders(k).name)
   
    AllIMU.PhoneIMUReadings{1,i}.stepDetector = stepDetector;
    AllIMU.PhoneIMUReadings{1,i}.walkingFrequency = walkingFrequency;
    
    PhoneIMUReadings{1,i} = AllIMU.PhoneIMUReadings{1,i};
    
   walkingS = find(AllIMU.PhoneIMUReadings{1,i}.stepDetector==1);

    AllIMU.PhoneIMUReadings{1,i}.PhoneID = replace(AllIMU.PhoneIMUReadings{1,i}.PhoneID," ","");
  
end

fname = subFolders(k).name+"IMU";
save(fname,'PhoneIMUReadings')
cd(src_path)
% end