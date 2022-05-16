clc;
clear;
close all
%% hilti: quaternion is xyzw, different from matlab
% livox Tlidar_to_imu
q_l2i =  [0.999993918507834, 0.0012283821413574625, -0.0032596475280467258, -0.00016947759535612024]; 
q_l2i = rev(q_l2i);
R_l2i = quat2rotm(q_l2i);
t_l2i = [-0.003050707070885951, -0.021993853931529066, 0.15076415229379997]';
% show(R_l2i,t_l2i);

% ouster
q_lO2i =  [0.999993918507834, 0.0012283821413574625, -0.0032596475280467258,   -0.00016947759535612024]; 
q_lO2i = rev(q_lO2i);
t_lO2i = [0.01001966915517371, -0.006645473484212856, 0.09473042428051345]';
R_lO2i = quat2rotm(q_lO2i);
show(R_lO2i,t_lO2i);

% imu_adis: Tlidar_to_imuAdis
q_iA2i = [-0.7046053008605138, 0.7095927549150899, 0.00202922313636401, -0.002318280556988682]; 
q_iA2i = rev(q_iA2i);
t_iA2i = [-0.028227032742698683, -0.006375545183463591, -0.03171966135034937]';
R_iA2i = quat2rotm(q_iA2i);
R_l2iA = R_iA2i'*R_l2i;
t_l2iA = R_iA2i'*(t_l2i-t_iA2i);
% show(R_l2iA,t_l2iA);

q_iO2i =[0.999995768781057, 0.0013912008656786696, -0.0016769584679952214,-0.0019273791610877828];
q_iO2i = rev(q_iO2i);
t_iO2i = [-0.012106872487123346, -0.01947953900975557, 0.11154467137366508]';
R_iO2i = quat2rotm(q_iO2i);
R_l2iO = R_iO2i'*R_l2i;
t_l2iO = R_iO2i'*(t_l2i-t_iO2i);
% show(R_l2iO,t_l2iO);

% cam0: T_c2i
q_c2i = [-0.5003218001035493, 0.5012125349997221, -0.5001966939080825, 0.49826434600894337];
q_c2i = rev(q_c2i);
t_c2i = [0.05067834857850693, 0.0458784339890185, -0.005943648304780761]';
R_c2i = quat2rotm(q_c2i);
% R_i2c = R_c2i;
% t_i2c = -R_c2i'*t_c2i;
show(R_c2i,t_c2i);

%% smart wheelchair
R_i2l = [-0.018979439340747, -0.018593533525231, -0.999646763493649;
                -0.063870582944317, -0.997762361026756, 0.019771064929604;
                -0.997777566533844, 0.064223274288635, 0.017749927467154];
t_i2l = [-0.053848665846961,0.022563392863635,0.041302891248130]';
% show(R_i2l',-R_i2l'*t_i2l);


%% Apollo South Bay extrensics 
% t_lidar2imu = [0.010524; 1.613485 ; 1.228857]
% quat_dxl = [-0.003936 0.007856 0.707845 0.706313]; % xyzw
% quat_dxl2 = [0.706313 -0.003936 0.007856 0.707845]; % wxyz
% R_lidar2imu = quat2rotm(quat_dxl2)
% R_imu2lidar = R_lidar2imu'
% t_imu2lidar = -R_imu2lidar*t_lidar2imu

% pcshow

% pc = pcread("/mnt/sdb/Datasets/NCLT/datasets/fastlio_loc2/2012-02-02-gt/map_pcd/errorMap.pcd");
% pcshow(pc)

function str = convert(R)
[n,m] = size(R);
str = "[";
for i = 1:n
    for j=1:m
        if i*j~=m*n
            str = str+num2str(R(i,j))+", ";
        else
            str = str+num2str(R(i,j))+"]";
        end
    end
end
end

function y = show(R,t)
    disp(convert(R));
    disp(convert(t));
end

function q2=rev(q)
    q2 = [q(4),q(1),q(2),q(3)];
end
