#ifndef KINECTFUSION_RECONSTRUCTION_H
#define KINECTFUSION_RECONSTRUCTION_H
#include "CPointCloud.h"
#include "Config.h"
#include "CudaFunctions.h"
#include "IOHelper.h"
#include "cxtimers.h"

#include "yaml-cpp/yaml.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "sophus/so3.hpp"
class KinectFusionReconstruction {
public:
    typedef Eigen::Matrix<hostComplex, 3, 3, Eigen::RowMajor> Matrix3frm;
    typedef Eigen::Matrix<hostComplex, 3, 3> Matrix3c;
    typedef Eigen::Matrix<hostComplex, 3, 1> Vector3c;
    typedef Eigen::AngleAxis<hostComplex> AngleAxisc;
    typedef Eigen::Matrix<hostComplexICP, 3, 1> Vector3cICP;
    typedef Eigen::Matrix<hostComplexICP, 3, 3> Matrix3cICP;

    YAML::Node config;
    cx::timer time_logger;
    int num_levels = 3;
    Eigen::Matrix4cf world2camera;
    Eigen::Matrix4cf world2volume;
    std::vector<Eigen::Matrix4cf> world2camera_record;
    int frame_id;///	frame id
    int frame_step;
    std::vector<Eigen::Matrix4f> gt_poses{};

    // camera related
    Intr kinect_intrinsic;        ///	intrinsic parameters of kinect
    int depth_width, depth_height;///	resolution of the depth map

    // TSDF volume
    Eigen::Vector3i volume_resolution;
    float voxel_size{};
    TsdfVolume *tsdf_volume_d_ptr;///	the volume
    int max_integration_weight;

    // ICP
    int icp_iterations[3]{};           ///	iterations of ICP each pyramid level
    float distThres{};                 ///	ICP inlier distance threshold
    float angleThres{};                ///	ICP inlier angle threshold
    DeviceArray2D<devComplexICP> g_buf;///	Temporary buffer for ICP
    DeviceArray<devComplexICP> sum_buf;///	Buffer to store MLS matrix
    DeviceArray2D<float> jacobi_buf;
    DeviceArray2D<float> hessian_buf[12];
    Eigen::Matrix4f jacobi_loss_pose;
    Eigen::Matrix4f *hessian_loss_pose_list[3];

    float trunc_logistic_k;// parameter for logistic trunc in tsdf fusion

    // map data
    float biInterpolate_threshold{0.005f};// parameter for bilinear interpolate in image space

    DeviceArray2D<float> newTSDF_volume;                  /// volume of newTSDF value in volume coordinate
    std::vector<DeviceArray2D<devComplex>> depths_curr_d; ///	filter depth pyramid
    std::vector<DeviceArray2D<devComplex>> vmaps_curr_d;  ///	pyramid of current point cloud in camera coordinate
    std::vector<DeviceArray2D<devComplex>> nmaps_curr_d;  ///	pyramid of current normal map in camera coordinate
    std::vector<DeviceArray2D<devComplex>> vmaps_g_prev_d;///	pyramid of previous point cloud
                                                          /// in global coordinate
    std::vector<DeviceArray2D<devComplex>> nmaps_g_prev_d;///	pyramid of previous normal map
                                                          /// in global coordinate
    DeviceArray2D<float> depthRawScaled_d;                ///	depthRawScaled Buffer for scaled depth along ray

    // parameters for grad LM
    hostComplex damp{1e-6, 0.0};
    hostComplex lambda_max{2.0, 0.0};
    hostComplex lambda_min{0.5, 0.0};
    floatType B1 = 1.0;
    floatType B2 = 1.0;

    // control flags
    bool use_gtPose;

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    int getFrame() const { return frame_id; }

    int getVolumeSize() const { return volume_resolution[0] * volume_resolution[1] * volume_resolution[2]; }

    Eigen::Matrix4cf getCamera2Volume() { return world2volume * world2camera.inverse(); }

    void saveTSDFVolume(const std::string& tsdf_filename);

    CPointCloud generateCurrPC(int level);

    CPointCloud generatePrevPC(int level);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // KinectFusion Pipeline Functions
    KinectFusionReconstruction();

    void AllocateBuffers();

    void ReleaseBuffers();

    // set parameter from config file
    void SetYamlParameters(const YAML::Node& config_);

    /*
      align the frame and fuse into volume
  */
    int ProcessFrame(const DeviceArray2D<ushort> &depth_frame_d);

    void SurfaceMeasure(const DeviceArray2D<ushort> &depth_frame_d);

    int PoseEstimate(Matrix3frm Rcurr, Eigen::Vector3cf tcurr,
                     Matrix3frm Rprev_inv, Eigen::Vector3cf tprev);

    int PoseNewtonEstimate(Matrix3frm Rcurr, Eigen::Vector3cf tcurr,
                           Matrix3frm Rprev_inv, Eigen::Vector3cf tprev);

    /*
      align the frame
  */
    int AlignDepthToReconstruction(const DeviceArray2D<ushort> &depth_frame_d,
                                   bool use_LM = false);

    /*
      integrate the frame into volume
  */
    int IntegrateFrame(const DeviceArray2D<ushort> &depth_frame_d);

    /*
      smooth the unsigned short depth map
  */
    static int SmoothDepthFrame(MapArr &dst_d,
                                const DeviceArray2D<ushort> &src_d);

    /*
      calculate point cloud from given view by ray casting
  */
    int CalculatePointCloud(MapArr &xyz_g_d, MapArr &normal_g_d);

    /*
  extract point cloud from tsdf volume by ray casting
  */
    CPointCloud ExportPointCloud(int max_buffer) const;


    float2 ComputeTSDF_loss(const Eigen::VectorXf &xi, const DeviceArray2D<ushort> &depth_frame_d,
                            thrustDvec<float> &gt_vec,
                            thrustDvec<float> &real_vec,
                            thrustDvec<int> &count_vec);


    float4 ComputeTSDF_hessian(const VectorXdc &xi, const DeviceArray2D<ushort> &depth_frame_d,
                               thrustDvec<float> &gt_vec,
                               thrustDvec<float> &real_vec,
                               thrustDvec<float> &grad_vec,
                               thrustDvec<float> &hessian_vec,
                               thrustDvec<int> &count_vec);

    static inline Eigen::Vector3cf
    GetTranslation(Eigen::Matrix4cf &trans_mat) {
        return trans_mat.block<3, 1>(0, 3);
    }

    static inline Eigen::Matrix3cf
    GetRotation(Eigen::Matrix4cf &trans_mat) {
        Matrix3frm R;
        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 3; i++)
                R(j, i) = trans_mat(j, i);
        return R;
    }

    static Eigen::Matrix4cf se3Exp(const Eigen::VectorXcf &xi) {
        if (xi.size() != 6) {
            printf("Error! The size of coordinate-vector xi in se(3) must be 6");
            exit(0);
        }
        Eigen::Vector3cf v = xi.head(3);
        Eigen::Vector3cf omega = xi.tail(3);
        Eigen::Matrix3cf omegaHat;
        omegaHat.setZero();
        omegaHat(0, 1) = -omega[2];
        omegaHat(0, 2) = omega[1];
        omegaHat(1, 2) = -omega[0];
        omegaHat(1, 0) = omega[2];
        omegaHat(2, 0) = -omega[1];
        omegaHat(2, 1) = omega[0];

        Eigen::Matrix3cf R, V;
        R.setIdentity();
        V.setIdentity();
        if (omega.norm() < 1e-6) {
            R += omegaHat;
            V += omegaHat;
        } else {
            std::complex<float> sum =
                    omega.transpose() *
                    omega;
            std::complex<float> theta = sqrt(sum);
            std::complex<float> s = sin(theta);
            std::complex<float> c = cos(theta);
            Eigen::Matrix3cf omega_hat_sq = omegaHat * omegaHat;
            std::complex<float> A = s / theta;
            std::complex<float> B = (1.0f - c) / pow(theta, 2.0f);
            std::complex<float> C = (theta - s) / pow(theta, 3.0f);
            R = R + A * omegaHat + B * omega_hat_sq;
            V = V + B * omegaHat + C * omega_hat_sq;
        }
        Eigen::Vector3cf t = V * v;
        Eigen::Matrix4cf xi_exp;
        xi_exp.setZero();
        xi_exp.block<3, 3>(0, 0) = R;
        xi_exp.block<3, 1>(0, 3) = t;
        xi_exp(3, 3) = 1;
        return xi_exp;
    }
};
#endif// KINECTFUSION_RECONSTRUCTION_H
