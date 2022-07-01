clc;
clear;
close all
startDate = "2012-01-15";
%% LOMAL
dateLists = ["2012-02-02","2012-03-17","2012-04-29",...
    "2012-05-11","2012-06-15","2012-08-04","2012-11-17","2013-01-10","2013-02-23"]';
folder = "/mnt/sdb/Datasets/NCLT/datasets/no_LIO";

% %% no TMM
% dateLists = ["2012-02-02","2012-03-17","2012-04-29","2012-05-11"]';
% folder = "/mnt/sdb/Datasets/NCLT/datasets/fastlio_noTMM";

% %% no CC
% dateLists = ["2012-02-02","2012-03-17","2012-04-29","2012-05-11"]';
% folder = "/mnt/sdb/Datasets/NCLT/datasets/fastlio_noCC";

setNum = length(dateLists);
daysPassed = cntDays(startDate,dateLists);

ateErrorCell = cell(setNum,1);
percWithinOneMeter = zeros(setNum,1);
percWithin05Meter = zeros(setNum,1);
percWithin02Meter = zeros(setNum,1);
percWithin01Meter = zeros(setNum,1);
stdError = zeros(setNum,1);
meanError = zeros(setNum,1);
maxError = zeros(setNum,1);
locFrequency =  zeros(setNum,1);
locFrequencyG =  zeros(setNum,1);
successRate =  zeros(setNum,1);
TMMno = zeros(setNum,1);
duration = zeros(setNum,1);
mapExtension = cell(setNum,1);

for iB=1:setNum
    date = dateLists{iB};
    disp("reading "+date);
    logFilePath = folder+"/"+date+"/map_pcd/mappingError.txt";
    poseFilePath = folder+"/"+date+"/map_pcd/path_mapping.txt";
%     poseFilePath = folder+"/"+date+"/map_pcd/path_fusion.txt"; 
%     poseFilePath = folder+"/"+date+"/map_pcd/path_vinsfusion.txt";
    gtFilePath = "/mnt/sdb/Datasets/NCLT/datasets/"+date+"/groundtruth_"+date+".csv";

    %% log file reading
    fID = fopen(logFilePath);
    logData = textscan(fID,strPatternGenerate(11));
    timeLog = logData{1}-logData{1}(1);
    regiError = logData{5};
    inlierRatio2 = logData{4};
    inlierRatio = logData{3};
    isTMM = logData{2};
    
    if any(inlierRatio2 <0.1)
        mapExtension{iB} = "YES";
    else
        mapExtension{iB} = "NO";
    end
    %% pose file reading
    fID2 = fopen(poseFilePath);
    poseData = textscan(fID2,strPatternGenerate(7));
    lenPose = length(poseData{1});
    matPose = zeros(lenPose,7);
    for i=1:lenPose
        for j=1:7
            matPose(i,j) = poseData{j}(i);
        end
    end

    %% gt reading
    % readcsv readmatrix:sth is wrong
    fID3 = fopen(gtFilePath);
    gtData = textscan(fID3, "%f%s%f%s%f%s%f%s%f%s%f%s%f");
    % downsample
    downsample = 10;
    lenGT = length(gtData{1});
    matGT = zeros(floor(lenGT/10),7);
    %% roll+fastlio uses imu pose, so here convert body pose to imu pose
    tbi = [-0.11 -0.18 -0.71]';
    for i=1:floor(lenGT/10)
        for j=1:7
            matGT(i,j) = gtData{2*j-1}(10*i);
        end
        Rmb = eul2rotm([ matGT(i,7),matGT(i,6),matGT(i,5)],"ZYX");
        tmb = matGT(i,2:4)';
        tmi = Rmb*tbi +tmb;
        matGT(i,2:4) = tmi';
    end
    %% sync with time
    timeGT = matGT(:,1)/1e+6; % us -> sec
    timePose =  matPose(:,1)/1e+6;
    MDtimeGT = KDTreeSearcher(timeGT);
    [idx, D] = rangesearch(MDtimeGT,timePose,0.05);
    ateError = zeros(lenPose,1);
    not_found = 0;

    for i=1:lenPose
        if isempty(idx{i})
            not_found = not_found + 1;
            continue;    
        end
        %% rule out obvious wrong ground truth, please check for yourself
        if date=="2013-02-23" && matPose(i,2)>-310 && matPose(i,2)<-260&&...
            matPose(i,3)>-450 && matPose(i,3)<-435
            continue;
        end        
        %% convert gt body to gt imu
        ateError(i) = norm(matPose(i,2:3)-matGT(idx{i}(1),2:3));
    end
    ateErrorCell{iB} = ateError;
    percWithinOneMeter(iB) = 100 -length(find(ateError> 1.0))/length(ateError)*100;
    percWithin05Meter(iB) = 100 -length(find(ateError> 0.5))/length(ateError)*100;
    percWithin02Meter(iB) = 100 -length(find(ateError> 0.2))/length(ateError)*100;
    percWithin01Meter(iB) = 100 -length(find(ateError> 0.1))/length(ateError)*100;
    stdError(iB) = std(ateError);
    meanError(iB) = mean(ateError);
    maxError(iB)= max(ateError);
    [TMMno(iB), positions] = TMMcnt(isTMM);
    locFrequency(iB) = lenPose/(timeGT(end)-timeGT(1));
    locFrequencyG(iB) = length(timeLog)/(timeGT(end)-timeGT(1));
    duration(iB) = (timeGT(end)-timeGT(1))/3600; 
    successRate(iB) = length(find(ateError < 1.0))/10/(timeGT(end)-timeGT(1))*100;
    

    figure(1)
    histogram(ateError,"DisplayStyle","stairs");
    hold on
    
%     figure(2)
%     idxOver1m = find(ateError>1.0);
%     plot(matPose(idxOver1m,2),matPose(idxOver1m,3),".","MarkerSize",10);
%     hold on
    
    figure(3)
    y = matPose(:,3);
    y(end) = nan;
    patch(matPose(:,2),y,ateError,'EdgeColor','interp','MarkerFaceColor','flat');
    hold on
    
    for iT = 1:TMMno(iB)
        plot(matPose(positions(iT),2),matPose(positions(iT),3),'o','MarkerFaceColor','r');
    end
    
end
colorbar;
% legend(dateLists);
hold off
figure(1)
legend(dateLists);
xlabel("Localization error (m)");
ylabel("Count");

figure(3)
xlabel('X (m)');
ylabel('Y (m)');
%% PLOT and SAVE
T = table(dateLists,daysPassed,duration,meanError,stdError,maxError,percWithin01Meter,percWithin02Meter,...
    percWithin05Meter,percWithinOneMeter,successRate, locFrequency,locFrequencyG,TMMno,mapExtension);
writetable(T,"results/loc_results_v4.xls");

%% overall mean and std
ateAll = [];
for i=1:setNum
    ateAll = [ateAll; ateError];
end
disp("mean error: "+ num2str(mean(ateAll)));
disp("std error: " + num2str(std(ateAll)));


% a=[timePose-timePose(1) ateError];
function strPattern =strPatternGenerate(n)
    strPattern = "";
    for i=1:n
        strPattern = strPattern+"%f";
    end

end

function [n, positions] = TMMcnt(x)
    n=0;
    positions = zeros(10,1);
    for i=1:length(x)-1
        if x(i+1) -x(i) > 0.9
            n = n+1;
            positions(n) = i;
        end
    end
end

function days = cntDays(start,str)
    n = length(str);
    days = zeros(n,1);
    day1 =convertNum(start) ;
    for i=1:n
        day2 = convertNum(str(i));
        days(i) = datenum(day2(1),day2(2),day2(3))-datenum(day1(1),day1(2),day1(3));        
    end
end
function nums = convertNum(x)
    strCell = split(x,"-");
    nums = zeros(3,1);
    for i=1:3
        nums(i) = strCell(i);
    end
end