#include <teaser/matcher.h>
#include <teaser/registration.h>
#include <pcl/features/normal_3d.h> // normal estimation

#include <pcl/features/fpfh_omp.h> //包含fpfh加速计算的omp(多核并行计算)

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include"tic_toc.h"

using namespace std;
class TEASERmanager
{
    public:
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_voxel_;
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_voxel_;

        float inlier_fraction_threshold_ = 0.5;
        float error_threshold_ = 0.2;
        float outlier_threshold_ = 1.0;
        
        float normal_estimation_radius_ = 2.0;
        float fpfh_radius_ = 2.0; // 2m is better than 4m
        float filter_size_ = 0.5;

        float matching_error;
        float inlier_fraction;
        Eigen::Isometry3f transformation;
        bool get_trans;

    // teaser needs no intensity channel
    TEASERmanager(pcl::PointCloud<pcl::PointXYZI>::Ptr sourceI,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr targetI)
    {
        // test if it crashes during run time !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        source_voxel_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        target_voxel_.reset(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());
        transformXYZItoXYZ(sourceI,source);
        transformXYZItoXYZ(targetI,target);

        vector<int> index;
        pcl::removeNaNFromPointCloud(*source, *source, index);
        pcl::removeNaNFromPointCloud(*target, *target, index);
        source_voxel_ = voxelGrid(source);
        target_voxel_ = voxelGrid(target);

        get_trans = false;

        transformation = Eigen::Isometry3f::Identity();

        cout<<"Source cloud size: "<<source_voxel_->size()<<" Target cloud size: "<<target_voxel_->size()<<endl;

    }
    void transformXYZItoXYZ(pcl::PointCloud< pcl::PointXYZI >::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud)
    {
        output_cloud->clear();
        for (int i = 0; i < int(input_cloud->size());i++){
            pcl::PointXYZ pt; 
            pt.x = input_cloud->points[i].x;
            pt.y = input_cloud->points[i].y;
            pt.z = input_cloud->points[i].z;
            output_cloud->push_back(pt);
        }
    }

    void setParameters( float inlier_fraction_threshold, float error_threshold, float outlier_threshold,
                        float normal_estimation_radius = 2.0,
                        float fpfh_radius = 2.0,
                        float filter_size = 0.5)
    {
        inlier_fraction_threshold_ = inlier_fraction_threshold;
        error_threshold_ = error_threshold;
        outlier_threshold_ = outlier_threshold;

        normal_estimation_radius_ = normal_estimation_radius;
        fpfh_radius_ = fpfh_radius;
        filter_size_ = filter_size;
    }

    Eigen::Isometry3f getTransformation()
    {
        TicToc teaser_match;
        if (source_voxel_->empty() || target_voxel_->empty()) 
        {
            get_trans = false;
            cout<<"Empty clouds!"<<endl;
            return transformation;
        }

        pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_fpfh = compute_fpfh_feature(source_voxel_);
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_fpfh = compute_fpfh_feature(target_voxel_);

        // convert to teaser-typed voxel & fpfh
        teaser::PointCloud target_voxel_T, source_voxel_T;
        teaser::FPFHCloud target_fpfhT, source_fpfhT;
        target_voxel_T.reserve(target_voxel_->size());
        target_fpfhT.reserve(target_voxel_->size());


        for (int i = 0; i < int(target_voxel_->size()); i++) {
            target_voxel_T.push_back({target_voxel_->at(i).x, target_voxel_->at(i).y, target_voxel_->at(i).z});
            target_fpfhT.push_back(target_fpfh->at(i));
        }

        source_voxel_T.reserve(source_voxel_->size());
        source_fpfhT.reserve(source_fpfh->size());
        for (int i = 0; i < int(source_voxel_->size()); i++) {
            source_voxel_T.push_back({source_voxel_->at(i).x, source_voxel_->at(i).y, source_voxel_->at(i).z});
            source_fpfhT.push_back(source_fpfh->at(i));
        }

        teaser::Matcher matcher;
        auto correspondences = matcher.calculateCorrespondences(
            source_voxel_T,
            target_voxel_T,
            source_fpfhT,
            target_fpfhT,
            false,
            false, // crosscheck
            false, // tuple check
            0.95); // tuple_scale

        teaser::RobustRegistrationSolver::Params params;
        params.noise_bound = 0.5;
        params.cbar2 = 1.0;
        params.estimate_scaling = false;
        params.rotation_max_iterations = 100;
        params.rotation_gnc_factor = 1.4;
        params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
        params.rotation_cost_threshold = 0.005;
        // params.v

        teaser::RobustRegistrationSolver solver(params);
        solver.solve(source_voxel_T, target_voxel_T, correspondences);

        auto solution = solver.getSolution();

        transformation.linear() = solution.rotation.cast<float>();
        transformation.translation() = solution.translation.cast<float>();

        cout<<"The solution matrix: "<<transformation.matrix()<<endl;

        cout << "Matching time is: " << teaser_match.toc() <<"ms"<< endl;

        get_trans = true;
        return transformation;
    }

    // make it compatible with other matching method?
    bool matched() 
    {
        if (get_trans)
        {   
            cout << "Getting error..."<<endl;
            TicToc get_error;
            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_target(new pcl::KdTreeFLANN<pcl::PointXYZ>());
            kdtree_target->setInputCloud(target_voxel_);

            vector<int> pointSearchInd;
            vector<float> pointSearchSqDis;
            int num_of_inliers = 0;
            float dist_mean = 0.0;
            for (auto & pt: source_voxel_->points)
            {
                kdtree_target->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
                if (pointSearchInd.empty()) continue;
                float dist = pointDistance(pt,target_voxel_->points[pointSearchInd[0]]);
                if (  dist < outlier_threshold_)
                {
                    num_of_inliers++;
                    dist_mean += dist;
                }
            }
            matching_error = dist_mean / num_of_inliers;
            inlier_fraction = num_of_inliers/source_voxel_->points.size();
            cout << "Time to get error is: " << get_error.toc() <<"ms"<< endl;
            cout<<"Matching error: " << matching_error << "****inlier fraction: "<<inlier_fraction<<endl;

            if(matching_error < error_threshold_ && inlier_fraction < inlier_fraction_threshold_) 
            {
                return true;
            }
            else
                return false;

        }
        else            
        {
            cout<<"No transformation yet!"<<endl;
            return false;
        }
    }

    float pointDistance(pcl::PointXYZ p1, pcl::PointXYZ p2)
    {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)
    {
        pcl::PointCloud<pcl::Normal>::Ptr point_normal(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> est_normal;
        est_normal.setInputCloud(input_cloud);

        est_normal.setRadiusSearch(normal_estimation_radius_); 
        est_normal.compute(*point_normal);//计算法向量

        //fpfh 估计
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh(new pcl::PointCloud<pcl::FPFHSignature33>);

        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> est_fpfh;
        est_fpfh.setNumberOfThreads(4); //指定4核计算

        est_fpfh.setInputCloud(input_cloud);
        est_fpfh.setInputNormals(point_normal);

        est_fpfh.setRadiusSearch(fpfh_radius_); 
        est_fpfh.compute(*fpfh);

        return fpfh;

    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> downSampled;  //创建滤波对象
        downSampled.setInputCloud(cloud_in);            //设置需要过滤的点云给滤波对象
        downSampled.setLeafSize(filter_size_, filter_size_, filter_size_); 
        downSampled.filter(*cloud_out);  //执行滤波处理，存储输出
        return cloud_out;
    }

        

};
