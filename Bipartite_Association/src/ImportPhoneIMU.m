function [PhoneID,PhoneIMUReadings] = ImportPhoneIMU(fname,IMUFiles,src_path,sequences_path,seqname)
%READ IMU DATA
for i=1:length(IMUFiles)
    newfname = strrep(IMUFiles(i).name,'/','\');
    PhoneIMU = importIMUfile(newfname);
    PhoneIMU.timestamp = PhoneIMU.timestamp./10^3;
    PhoneIMU.timestamp = datetime(PhoneIMU.timestamp,'ConvertFrom','posixtime','TimeZone','America/New_York','Format', 'MM-dd-yyyy HH:mm:ss.SSSSSS');
    PhoneIMU = table2timetable(PhoneIMU);
    Phonelin = PhoneIMU(PhoneIMU.tag == "LINEAR",1:4);
    Phonelin.tag = [];
    Phoneaccel = PhoneIMU(PhoneIMU.tag == "ACCEL",1:4);
    Phoneaccel.tag = [];
    Phoneaccel.gx = smooth(Phoneaccel.x,21, 'sgolay',1);
    Phoneaccel.gy = smooth(Phoneaccel.y,21, 'sgolay',1);
    Phoneaccel.gz = smooth(Phoneaccel.z,21, 'sgolay',1);
    x = Phoneaccel.x.^2;
    y = Phoneaccel.y.^2;
    z = Phoneaccel.z.^2;
    m = sqrt(x+y+z);
    Phoneaccel.gm = m;
    Phoneaccel.gm = Phoneaccel.gm - mean(Phoneaccel.gm);
    medfiltgm = medfilt1(Phoneaccel.gm,101);
    Phoneaccel.medfiltgm = medfiltgm;
	
    
    %READ GYROSCOPE DATA
    Phonegyro = PhoneIMU(PhoneIMU.tag == "GYRO",1:4);
    Phonegyro.tag = [];
    Phonegyro.x = rad2deg(Phonegyro.x);
    Phonegyro.y = rad2deg(Phonegyro.y);
    Phonegyro.z = rad2deg(Phonegyro.z);
    gyx = Phonegyro.x.^2;
    gyy = Phonegyro.y.^2;
    gyz = Phonegyro.z.^2;
    Phonegyro.mg = sqrt(gyx+gyy+gyz);


    %READ MAGNETOMETER DATA
    Phonemag = PhoneIMU(PhoneIMU.tag == "MAG",1:4);
    Phonemag.tag = [];
	
	
    IMUReadings = synchronize(Phoneaccel,Phonegyro,Phonemag,'union','nearest');
	
    IMUReadings = retime(IMUReadings,'regular','linear','SampleRate',50);
    PhoneID{1,i} = string(erase(IMUFiles(i).name,'0'));
    PhoneID{1,i} = replace(PhoneID{1,i},'.csv','');
    disp(PhoneID{1,i})
    IMUReadings.PhoneID(:) = PhoneID{1,i};
	
	
    currentPwd = pwd;
    cd(src_path);
	
    
    [yaw,pitch,roll] = getPhoneOrientation([IMUReadings.x_Phoneaccel...
        IMUReadings.y_Phoneaccel...
        IMUReadings.z_Phoneaccel],...
        [IMUReadings.x_Phonegyro...
        IMUReadings.y_Phonegyro...
        IMUReadings.z_Phonegyro],...
        [IMUReadings.x_Phonemag...
        IMUReadings.y_Phonemag...
        IMUReadings.z_Phonemag]);

    IMUReadings.yaw = yaw;
    IMUReadings.pitch = pitch;
    IMUReadings.roll = roll;
	
    PhoneIMUReadings{1,i} = IMUReadings;

    cd(sequences_path+"/"+seqname+"/");
	 
end

