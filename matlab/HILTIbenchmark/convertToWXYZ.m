clc
clear;
close all
sequence = "uzh_tracking_area_run2";
gtFilePath = "/mnt/sdb/Datasets/HILTI_dataset/"+sequence +"/"+sequence+".txt";
gtFilePath2 = "/mnt/sdb/Datasets/HILTI_dataset/"+sequence +"/"+sequence+"wxyz.txt";
%% gt reading
% readcsv readmatrix:sth is wrong
fID3 = fopen(gtFilePath);
strPattern = pattern(8);
gtData = textscan(fID3, strPattern,"Headerlines",1);
lenPose = length(gtData{1});
%% writing
fID = fopen(gtFilePath2,'w');
for i = 1:lenPose
    fprintf(fID,"%f %f %f %f %f %f %f %f\n",[gtData{1}(i),gtData{2}(i),gtData{3}(i),gtData{4}(i),...
        gtData{8}(i),gtData{5}(i),gtData{6}(i),gtData{7}(i)]);
end
function s = pattern(n)
    s="";
    for i=1:n
        s = s+"%f";
    end
end