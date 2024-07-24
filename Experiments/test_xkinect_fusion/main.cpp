#include "Dataset.h"
#include "KinectFusionReconstruction.h"

#include "yaml-cpp/yaml.h"
#include <iostream>
#include <filesystem>

void savePose(const std::string &output_dir, int frame_id, const Eigen::Matrix4f &pose)
{
    std::stringstream ss;
    ss << "frame-" << std::setw(6) << std::setfill('0') << frame_id << ".pose.txt";
    std::string file_name = output_dir + ss.str();
    saveTxtMatrix(file_name, pose);
}

int main(int argc, char *argv[])
{
    std::cout << "Demo of XKinectFusion" << std::endl;
    if (argc < 2)
    {
        std::cout << "please enter the config file name"
                  << "\n";
        return -1;
    }
    const char *config_filename = argv[1];
    YAML::Node config = YAML::LoadFile(config_filename);
    Dataset dataset;
    auto dataset_format = config["dataset_format"].as<std::string>();
    auto dataset_dir = config["dataset_dir"].as<std::string>();
    auto start_frame = config["start_frame"].as<int>();
    auto end_frame = config["end_frame"].as<int>();
    auto is_flip = config["is_flip"].as<bool>();
    auto output_path = config["output_dir"].as<std::string>();
    ////////////////////////////////////////
    dataset = ICL_Dataset(dataset_dir, start_frame, end_frame, is_flip);
    std::cout << "frame num: " << dataset.size() << std::endl;
    ///////////////////////////////////////
    std::cout << "initialize kinect fusion......" << std::endl;
    KinectFusionReconstruction kinfu;
    kinfu.SetYamlParameters(config);
    //////////////////////////////////////
    cx::timer time_logger;
    double total_time = 0;
    std::cout << "start slam!" << std::endl;

    while (kinfu.frame_id < end_frame)
    {
        int frame_id = kinfu.frame_id;
        std::cout << "current frame is " << frame_id << "\n";
        cv::Mat depth_map, depth_show;
        dataset.getDepthData(frame_id, depth_map);
        auto *depth_ptr = depth_map.ptr<unsigned short>(0);
        DeviceArray2D<ushort> depth_frame_d;
        depth_frame_d.upload(depth_ptr, kinfu.depth_width * sizeof(ushort),
                             kinfu.depth_height, kinfu.depth_width);
        // c. process kinect fusion
        time_logger.reset();
        kinfu.ProcessFrame(depth_frame_d);
        double frame_time = time_logger.lap_ms();
        total_time += frame_time;
        Eigen::Matrix4f pose_c2w = kinfu.world2camera_record.back().inverse().real();
        Eigen::Matrix4cf pose_c2v = kinfu.getCamera2Volume();
        if (config["log_slam_pose"].as<bool>())
        {
            std::filesystem::create_directories((output_path + "slam/"));
            std::cout << "slam c2w:\n " << pose_c2w << std::endl;
            savePose(output_path + "slam/", frame_id, pose_c2w);
        }
        if (config["log_gt_pose"].as<bool>())
        {
            std::filesystem::create_directories((output_path + "gt/"));
            Eigen::Matrix4f gt_pose_c2w = dataset.getPose(0).inverse() * dataset.getPose(frame_id);
            std::cout << "gt c2w:\n " << gt_pose_c2w << std::endl;
            savePose(output_path + "gt/", frame_id, gt_pose_c2w);
        }
        if (config["draw_pcd"].as<bool>())
        {
            CPointCloud pc = kinfu.ExportPointCloud(1000000);
            if (frame_id == end_frame - 1)
                pc.exportPly(output_path + "pcd.ply");
        }
    }
    printf("mean frame time = %.3f ms\n", total_time / kinfu.getFrame());
}