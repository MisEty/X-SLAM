//
// Created by MiseTy on 2023/1/31.
//

#ifndef CSFD_SLAM_DATASET_H
#define CSFD_SLAM_DATASET_H

#include "IOHelper.h"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class Dataset
{
public:
    explicit Dataset(bool is_flip = false) { is_flip_ = is_flip; }
    int size() const { return size_; }

    // read depth data from CV_16U image
    void getDepthData(int index, cv::Mat &depth);

    //  // read depth data from CV_16U image
    //  void getDepthData_raw(int index, cv::Mat &depth);

    Eigen::Matrix4f getPose(int index)
    {
        assert(0 <= index && index < size_);
        return gt_poses_[index];
    }

    std::vector<Eigen::Matrix4f> getAllPose()
    {
        return gt_poses_;
    }

    void setPose(int index, Eigen::Matrix4f pose)
    {
        assert(0 <= index && index < size_);
        gt_poses_[index] = std::move(pose);
    }

    std::string getTimestamp(int index)
    {
        assert(0 <= index && index < size_);
        return time_stamps_[index];
    }

protected:
    int size_{}; // the number of images
    bool is_flip_;
    int factor_ = 1;
    std::vector<std::string> depths_filenames_;
    std::vector<Eigen::Matrix4f> gt_poses_;
    std::vector<std::string> time_stamps_;
};

class seven_scenes_Dataset : public Dataset
{
public:
    std::vector<Eigen::Matrix4f> init_poses_;

    explicit seven_scenes_Dataset(std::string &dataset_dir, std::vector<int> start_frames, std::vector<int> end_frames,
                                  std::vector<std::string> seq_names, bool is_flip = false);

    static void readInfo(std::string &filename, std::vector<int> &start_frames,
                         std::vector<int> &end_frames,
                         std::vector<std::string> &seq_names);
};

class ICL_Dataset : public Dataset
{
public:
    explicit ICL_Dataset(std::string &dataset_dir, int start_frame, int end_frame, bool is_flip = false);

    static bool readPoseFile(const std::string &poses_path, int start, int end, Eigen::Matrix4f &pose);
};

#endif // CSFD_SLAM_DATASET_H
