function [PhoneID,PhoneIMUReadings] = ImportPhoneIMU(fname,IMUFiles,src_path,sequences_path,seqname)
%READ IMU DATA
for i=1:length(IMUFiles)
    newfname = strrep(IMUFiles(i).name,'/','\');
    PhoneIMU = importIMUfile(newfname);
    PhoneIMU.timestamp = PhoneIMU.timestamp./10^3;
    PhoneIMU.timestamp = datetime(PhoneIMU.timestamp,'ConvertFrom','posixtime','TimeZone','America/New_York','Format', 'MM/dd/yy HH:mm:ss.SSSSSS');
    %     PhoneIMU.timestamp.Day = 23;
    %     PhoneIMU.timestamp.Year = 2020;
    %     PhoneIMU.timestamp.Month = 12;
    PhoneIMU = table2timetable(PhoneIMU);

    %READ Linear DATA
    Phonelin = PhoneIMU(PhoneIMU.tag == "LINEAR",1:4);
    Phonelin.tag = [];
    %     Phonelin = timetable2table(Phonelin);
    %READ ACCELEROMETER DATA
    Phoneaccel = PhoneIMU(PhoneIMU.tag == "ACCEL",1:4);
    Phoneaccel.tag = [];

    PhoneGPS = PhoneIMU(PhoneIMU.tag == "GPS",1:3);
    PhoneGPS.tag = [];
    %     Phoneaccel = timetable2table(Phoneaccel);
    %
    % Smoothing accelrometer signal
    Phoneaccel.gx = smooth(Phoneaccel.x,21, 'sgolay',1);
    Phoneaccel.gy = smooth(Phoneaccel.y,21, 'sgolay',1);
    Phoneaccel.gz = smooth(Phoneaccel.z,21, 'sgolay',1);
    %     Phoneaccel.gx = detrend(Phoneaccel.gx,5);
    %     Phoneaccel.gy = detrend(Phoneaccel.gy,5);
    %     Phoneaccel.gz = detrend(Phoneaccel.gz,5);

    %Get Acceleration Magnitude
    x = Phoneaccel.x.^2;
    y = Phoneaccel.y.^2;
    z = Phoneaccel.z.^2;
    m = sqrt(x+y+z);
    Phoneaccel.gm = m;
    %     Phoneaccel.gm = smooth(Phoneaccel.gm,21, 'sgolay',1);
    Phoneaccel.gm = Phoneaccel.gm - mean(Phoneaccel.gm);
    %     Phoneaccel.gm = smoothdata(Phoneaccel.gm,'rlowess',60);
    medfiltgm = medfilt1(Phoneaccel.gm,101);
    Phoneaccel.medfiltgm = medfiltgm;

    % subtracting gravity
    %     Phoneaccel.gx = Phoneaccel.gx - mean(Phoneaccel.gx);
    %     Phoneaccel.gy = Phoneaccel.gy - mean(Phoneaccel.gy);
    %     Phoneaccel.gz = Phoneaccel.gz - mean(Phoneaccel.gz);



    %READ Step Detector DATA
    Phonestep = PhoneIMU(PhoneIMU.tag == "STEPDET",1:2);
    Phonestep.tag = [];

    %Get Quaternion
    PhoneQuat = PhoneIMU(PhoneIMU.tag == "Quaternion",1:5);
    PhoneQuat.tag = [];


    %READ GYROSCOPE DATA
    Phonegyro = PhoneIMU(PhoneIMU.tag == "GYRO",1:4);
    Phonegyro.tag = [];
    %     Phonegyro = timetable2table(Phonegyro);

    Phonegyro.x = rad2deg(Phonegyro.x);
    Phonegyro.y = rad2deg(Phonegyro.y);
    Phonegyro.z = rad2deg(Phonegyro.z);

    %     Phonegyro.x = smooth(Phonegyro.x,21, 'sgolay',1);
    %     Phonegyro.y = smooth(Phonegyro.y,21, 'sgolay',1);
    %     Phonegyro.z = smooth(Phonegyro.z,21, 'sgolay',1);
    Phonegyro.x = smooth(Phonegyro.x,100, 'moving');
    Phonegyro.y = smooth(Phonegyro.y,100, 'moving');
    Phonegyro.z = smooth(Phonegyro.z,100, 'moving');

    gyx = Phonegyro.x.^2;
    gyy = Phonegyro.y.^2;
    gyz = Phonegyro.z.^2;
    Phonegyro.mg = sqrt(gyx+gyy+gyz);
    %READ MAGNETOMETER DATA
    Phonemag = PhoneIMU(PhoneIMU.tag == "MAG",1:4);
    Phonemag.tag = [];
    %     Phonemag = timetable2table(Phonemag);

    IMUReadings = synchronize(Phoneaccel,Phonegyro,Phonemag,'union','nearest');
    %     IMUReadings = synchronize(Phoneaccel,Phonestep,Phonegyro,Phonemag,PhoneQuat);

    PhoneID{1,i} = string(erase(IMUFiles(i).name,'0'));
    PhoneID{1,i} = replace(PhoneID{1,i},'.csv','');
    disp(PhoneID{1,i})
    IMUReadings.PhoneID(:) = PhoneID{1,i};
    currentPwd = pwd;
    cd(src_path);
    % apply magnatic declination offset:
    %-12.45421	0.03459	0.37478
    [yaw,pitch,roll] = getPhoneOrientation([IMUReadings.x_Phoneaccel...
        IMUReadings.y_Phoneaccel...
        IMUReadings.z_Phoneaccel],...
        [IMUReadings.x_Phonemag...
        IMUReadings.y_Phonemag...
        IMUReadings.z_Phonemag]);
    cd(sequences_path+"/"+seqname+"/");
    IMUReadings.yaw = medfilt1(yaw,200);
    IMUReadings.pitch = medfilt1(pitch,200);
    IMUReadings.roll = medfilt1(roll,200);

    %     IMUReadings = synchronize(Phonestep,IMUReadings3,'union','nearest');


    %     dt = milliseconds(100);
    %     IMUReadings = retime(IMUReadings,'regular','nearest','TimeStep',dt);
    PhoneIMUReadings{1,i} = IMUReadings;
    imutable = table(IMUReadings.timestamp,IMUReadings.x_Phonegyro,IMUReadings.y_Phonegyro,...
        IMUReadings.z_Phonegyro,IMUReadings.yaw);
    % writetable(imutable,fname+IMUReadings.PhoneID(1)+".csv",'delimiter',',')
end

