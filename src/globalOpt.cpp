/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

// adapted for global loc

#include "globalOpt.h"
#include "Factors.h"
double sta[6];

GlobalOptimization::GlobalOptimization(int maxNo, float ratio)
{
	initGPS = false;
    newGPS = false;
    newGlobalLocPose = false;
	WGlobal_T_WLocal = Eigen::Matrix4d::Identity();
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
    maxFrameNum = maxNo;
    globalLocalRatio = ratio;
    
}

GlobalOptimization::~GlobalOptimization()
{
    threadOpt.detach();
}

void GlobalOptimization::statistics(const map<double,double> m, double &mean, double &std)
{
    int len = m.size();
    if (m.empty()) return;
    for (auto i:m)
        mean += i.second;
    mean = mean/len;
    for(auto i:m)
    {
        std += (i.second-mean)*(i.second-mean);
    }
    if (len == 1) {
        std = 0;
        return;
    }
    std = sqrt(std/(len-1));
}
void GlobalOptimization::affine2qt(const Eigen::Matrix4d aff, Eigen::Vector3d &p, Eigen::Quaterniond &q)
{
    q = Eigen::Quaterniond(aff.block<3,3>(0,0));
    p = aff.block<3,1>(0,3);    
    // cout<<"q: "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
    return;
}

// here we update globally aligned odom pose asap
void GlobalOptimization::inputOdom(double t, Eigen::Matrix4d affine)
{
    Eigen::Vector3d OdomP;
    Eigen::Quaterniond OdomQ;
    affine2qt(affine,OdomP,OdomQ);

	mPoseMap.lock();
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(), 
    					     OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;


    Eigen::Quaterniond globalQ;
    globalQ = WGlobal_T_WLocal.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGlobal_T_WLocal.block<3, 3>(0, 0) * OdomP + WGlobal_T_WLocal.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose; 
    lastP = globalP;
    lastQ = globalQ;

    // sliding-window
    if ((int)localPoseMap.size() > maxFrameNum)
    {
        // special note: erase function of map is not the same as vector
        localPoseMap.erase((localPoseMap.begin())->first);
        globalPoseMap.erase((globalPoseMap.begin())->first);
    }

    mPoseMap.unlock();
}

void GlobalOptimization::getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

void GlobalOptimization::getTgl(Eigen::Matrix4d &tgl)
{
    tgl = WGlobal_T_WLocal;
}


void GlobalOptimization::inputGlobalLocPose(double t, Eigen::Matrix4d affine, double t_error, double q_error)
{
    mPoseMap.lock();
    Eigen::Vector3d locP;
    Eigen::Quaterniond locQ;
    affine2qt(affine,locP,locQ);
    vector<double> globalLocPose{locP.x(), locP.y(), locP.z(),
                              locQ.w(), locQ.x(), locQ.y(), locQ.z(), t_error, q_error};
    globalLocPoseMap[t] = globalLocPose;
    // vector<double> globalLocPose{locP.x(), locP.y(), locP.z(),t_error};
    // globalLocPoseMap[t] = globalLocPose; 

    if ((int)globalLocPoseMap.size() > globalLocalRatio*maxFrameNum) // global loc frequency is around 1/3 of fastlio odometry
    {
        globalLocPoseMap.erase((globalLocPoseMap.begin())->first);
    }
    newGlobalLocPose = true;
    mPoseMap.unlock();
}

void GlobalOptimization::optimize()
{
    while(true)
    {
        if(newGlobalLocPose)
        {
            newGlobalLocPose = false;
            // printf("global optimization using global localization and odometry\n");
            TicToc globalOptimizationTime;

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(1.0);
            ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

            //add param
            mPoseMap.lock();

            int length = localPoseMap.size();
            // w^t_i   w^q_i
            double t_array[length][3];
            double q_array[length][4];
            map<double, vector<double>>::iterator iter;
            iter = globalPoseMap.begin();  //using odomTOmap value for initial guess 

            // optimize the poses at a gps frequency
            for (int i = 0; i < length; i++, iter++)
            {
                t_array[i][0] = iter->second[0];
                t_array[i][1] = iter->second[1];
                t_array[i][2] = iter->second[2];
                q_array[i][0] = iter->second[3];
                q_array[i][1] = iter->second[4];
                q_array[i][2] = iter->second[5];
                q_array[i][3] = iter->second[6];
                // cout<<"q_array[i] "<<q_array[i][0]<<" "<<q_array[i][1]<<" "<<q_array[i][2]<<" "<<q_array[i][3]<<endl;
                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
            }

            // cout<<" pose no. before opt "<< length<<endl;
            map<double, vector<double>>::iterator iterVIO, iterVIONext, iterGlobalLoc;
            int i = 0;
            // int outlierNO = 0;
            for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++)
            {
                //vio factor
                iterVIONext = iterVIO;
                iterVIONext++;
                if(iterVIONext != localPoseMap.end())
                {
                    Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();
                    wTi.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4], 
                                                               iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                    wTi.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    wTj.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4], 
                                                               iterVIONext->second[5], iterVIONext->second[6]).toRotationMatrix();
                    wTj.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                    Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                    Eigen::Quaterniond iQj;
                    iQj = iTj.block<3, 3>(0, 0);
                    Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);

                    ceres::CostFunction* vio_function = RelativeRTError::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                                iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                                0.1, 0.01);
                    problem.AddResidualBlock(vio_function, NULL, q_array[i], t_array[i], q_array[i+1], t_array[i+1]);

                }
                
                double t = iterVIO->first;
                iterGlobalLoc = globalLocPoseMap.find(t); // synchronized global loc pose
                if (iterGlobalLoc != globalLocPoseMap.end())
                {
                    // if (length > 99)
                    // {
                    //     // get rid of outliers: not working, too many outliers
                    //     Eigen::Matrix4d odomPose = Eigen::Matrix4d::Identity(); 
                    //     Eigen::Matrix4d globalPose = Eigen::Matrix4d::Identity();
                    //     odomPose.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], 
                    //                                     localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                    //     odomPose.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                    //     globalPose.block<3, 3>(0, 0) = Eigen::Quaterniond(iterGlobalLoc->second[0], iterGlobalLoc->second[1], 
                    //                                                     iterGlobalLoc->second[2], iterGlobalLoc->second[3]).toRotationMatrix();
                    //     globalPose.block<3, 1>(0, 3) = Eigen::Vector3d(iterGlobalLoc->second[4],iterGlobalLoc->second[5],iterGlobalLoc->second[6]);
                    //     Eigen::Matrix4d Tgl = globalPose * odomPose.inverse();

                    //     if (fabs(Tgl(0,3)-sta[0])>3*sta[1] || fabs(Tgl(1,3)-sta[2])>3*sta[3] || fabs(Tgl(2,3)-sta[4])>3*sta[5])
                    //     {
                    //         outlierNO++;
                    //         continue;
                    //     }
                    // }

                        
                    // cout<<"found corresponding global loc pose "<<endl;
                    ceres::CostFunction* global_loc_function = globalTError::Create(iterGlobalLoc->second[0], iterGlobalLoc->second[1], 
                                                                       iterGlobalLoc->second[2], iterGlobalLoc->second[3],
                                                                       iterGlobalLoc->second[4],iterGlobalLoc->second[5],iterGlobalLoc->second[6],
                                                                       iterGlobalLoc->second[7],iterGlobalLoc->second[8]);
                    
                    problem.AddResidualBlock(global_loc_function, loss_function, q_array[i],t_array[i]);

                    // ceres::CostFunction* global_loc_function = TError::Create(iterGlobalLoc->second[0], iterGlobalLoc->second[1], 
                    //                                                    iterGlobalLoc->second[2], iterGlobalLoc->second[3]);
                    
                    // problem.AddResidualBlock(global_loc_function, loss_function, t_array[i]);
                }

            }
            
            ceres::Solve(options, &problem, &summary);
            // std::cout << summary.BriefReport() << "\n";

            // update global pose
            iter = globalPoseMap.begin();
            map<double,double> xGL;
            map<double,double> yGL;
            map<double,double> zGL;
            
            // Eigen::Matrix4d backup = WGlobal_T_WLocal;
            for (int i = 0; i < length; i++, iter++)
            {
            	vector<double> globalPose{t_array[i][0], t_array[i][1], t_array[i][2],
            							  q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
            	iter->second = globalPose;
                // only output the lastest
            	// if(i == length - 1)
            	// {
            	//     Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity(); 
            	//     Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
            	//     double t = iter->first;
            	//     WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], 
            	//                                                        localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
            	//     WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
            	//     WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4], 
            	//                                                         globalPose[5], globalPose[6]).toRotationMatrix();
            	//     WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
            	//     WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();
            	// }
                Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity(); 
                Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                double t = iter->first;
                WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], 
                                                                    localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4], 
                                                                    globalPose[5], globalPose[6]).toRotationMatrix();
                WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();
                xGL[t] = WGlobal_T_WLocal(0,3);
                yGL[t] = WGlobal_T_WLocal(1,3);
                zGL[t] = WGlobal_T_WLocal(2,3);
            }
            
            statistics(xGL,sta[0],sta[1]);
            statistics(yGL,sta[2],sta[3]);
            statistics(zGL,sta[4],sta[5]);
            xGL.clear();
            yGL.clear();
            zGL.clear();

            // cout<<"Tgl: "<<WGlobal_T_WLocal<<endl;
            // cout<<"mean x: "<<sta[0]<<" std x: "<<sta[1]<<endl;
            // cout<<"mean y: "<<sta[2]<<" std y: "<<sta[3]<<endl;
            // cout<<"mean z: "<<sta[4]<<" std z: "<<sta[5]<<endl;
            // double stdGL = sqrt(sta[1]*sta[1]+sta[3]*sta[3]+sta[5]*sta[5]);
            // cout<<"outlier num: "<<outlierNO<<endl;
            // cout<<"translation std: "<<stdGL<<endl;
            // cout<<"global time "<<globalOptimizationTime.toc()<<" ms"<<endl;
            // cout<<"pose size after opt:"<<globalPoseMap.size()<<endl;

            // if (stdGL > 1.0)                WGlobal_T_WLocal = backup;
            mPoseMap.unlock();
        }
        // waiting for poses to be accumulated
        std::chrono::milliseconds dura(1000);
        std::this_thread::sleep_for(dura);
    }
	return;
}

