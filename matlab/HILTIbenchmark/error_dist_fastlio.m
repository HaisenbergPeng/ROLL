clc
clear;
close all
%% Umeyama's method: how to align???
sequence = "test0202";
poseFilePath = "/home/haisenberg/Documents/ROLL/src/FAST_LIO/Log/"+sequence+"/mat_out.txt";
gtFilePath = "/mnt/sdb/Datasets/HILTI_dataset/Construction_Site_2_prism.txt";
%% log file reading
fID = fopen(poseFilePath);
strPattern = pattern(26);
logData = textscan(fID,strPattern);
matPose =[logData{1},logData{5},logData{6},logData{7},pi/180*logData{2},pi/180*logData{3},pi/180*logData{4}];
lenPose = length(matPose(:,1));

%% gt reading
% readcsv readmatrix:sth is wrong
fID3 = fopen(gtFilePath);
strPattern2 = pattern(8);
gtData = textscan(fID3, strPattern2,"Headerlines",1);
% downsample
downsample = 1;
lenGT = length(gtData{1});
matGT = zeros(lenGT,7);
%% hilti use imu frame but the first frame is not identity, so convert it
R_12m=eye(3); % convert the 1st frame to map
t_12m=zeros(3,1);
for i=1:floor(lenGT)
    if i==1
        R_12m = quat2rotm([gtData{8}(i),gtData{5}(i),gtData{6}(i),gtData{7}(i)]); % matlab: RzRyRx
        t_12m = [gtData{2}(i),gtData{3}(i),gtData{4}(i)]';
    else
        R_i2m= quat2rotm([gtData{8}(i),gtData{5}(i),gtData{6}(i),gtData{7}(i)]); % matlab: RzRyRx
        t_i2m = [gtData{2}(i),gtData{3}(i),gtData{4}(i)]';
        [R,t] = transformRT([R_12m t_12m;0 0 0 1],[R_i2m t_i2m;0 0 0 1]);
        tmp = rotm2eul(R,"ZYX"); % matlab: RzRyRx
        matGT(i,5:7) = [tmp(3),tmp(2),tmp(1)]; % To : roll pitch yaw
        matGT(i,2:4) = t;
    end
    matGT(i,1) = gtData{1}(i);
end

%% sync with time
timeGT = (matGT(:,1)-matGT(1,1)); % us -> sec
timePose = (matPose(:,1)-matPose(1,1)); % sec -> sec

% MDtimeGT = KDTreeSearcher(timeGT);
[idx, D] = rangesearch(timeGT,timePose,0.05);
ateErrorINI = zeros(lenPose,1);
Err = zeros(lenPose,6);
not_found = 0;
idxC = 0;
for i=1:lenPose
    if isempty(idx{i})
        not_found = not_found + 1;
        continue;    
    end
    idxC = idxC + 1;
%     ateError(i) = norm(matPose(i,2:3)-matGT(idx{i}(1),2:3));
    Err(i,:) = matPose(i,2:7)-matGT(idx{i}(1),2:7);
    for iE = 1:3
        Err(i,iE+3) = 180/pi*(Err(i,iE+3) - 2*pi*round(Err(i,iE+3)/2/pi));
    end
    deltaT = transError(matGT(idx{i}(1),2:7),matPose(i,2:7));
    ateErrorINI(idxC) = norm(deltaT(1:3,4));

end
ateError = ateErrorINI(1:idxC);

disp("RMSE error: "+norm(ateError)/sqrt(idxC))
disp("max error: "+max(ateError))
disp("Loc rate: "+length(ateError)/(timePose(end)-timePose(1)))
disp("Success ratio: "+100*length(find(ateError < 1.0))/(timeGT(end)-timeGT(1))/10)
disp("<0.1 %: "+ 100*length(find(ateError < 0.1))/idxC)
disp("<0.2 %: "+ 100*length(find(ateError < 0.2))/idxC)
disp("<0.5 %: "+ 100*length(find(ateError < 0.5))/idxC)
disp("<1.0 %: "+ 100*length(find(ateError < 1.0))/idxC)

figure(1)
set(gcf, "Position",[400,100,1000,800]);
hold on
plot(matPose(:,2),matPose(:,3),'--',"LineWidth",1.5);
plot(matGT(:,2),matGT(:,3),"LineWidth",1.5);
% legend("LOAM(M)+TM","LOAM(M)","ROLL","G.T.");
legend("fastlio","G.T.");
xlabel("X (m)");
ylabel("Y (m)");
saveas(1,sequence+".jpg");

function eT = transError(Vgt,V2)
% input: x y z r p y
    T1 = eye(4);
    T2 = eye(4);
    T1(1:3,1:3) = eul2rotm([Vgt(6),Vgt(5),Vgt(4)],"ZYX");
    T2(1:3,1:3) = eul2rotm([V2(6),V2(5),V2(4)],"ZYX");
    T1(1:3,4) = Vgt(1:3)';
    T2(1:3,4) = V2(1:3)';
    eT = T1\T2;    % equals to T1\T2
end

function y = removeJump(x)
    % 2pi
    Ntmp = round(x/360);
    y = x - Ntmp*360;
end
function s = pattern(n)
    s="";
    for i=1:n
        s = s+"%f";
    end
end

function [R,t]=transformRT(T1,T2)
    T = inv(T1)*T2;
    R = T(1:3,1:3);
    t= T(1:3,4);
end