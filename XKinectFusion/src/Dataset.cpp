#include "Dataset.h"

void Dataset::getDepthData(int index, cv::Mat &depth)
{
    assert(0 <= index && index < size_);
    //    std::cout << depths_filenames_[index] <<std::endl;
    depth = cv::imread(depths_filenames_[index], cv::IMREAD_UNCHANGED);
    depth /= factor_;
    if (is_flip_)
        cv::flip(depth, depth, 1);
}

seven_scenes_Dataset::seven_scenes_Dataset(std::string &dataset_dir, std::vector<int> start_frames,
                                           std::vector<int> end_frames, std::vector<std::string> seq_names, bool is_flip)
{
    int count = 0;
    int seq_num = seq_names.size();
    is_flip_ = is_flip;
    for (int seq = 0; seq < seq_num; seq++)
    {
        int start_frame = start_frames[seq];
        int end_frame = end_frames[seq];
        std::string seq_name = seq_names[seq];
        for (int frame = start_frame; frame <= end_frame; frame++)
        {
            std::string unformat = std::to_string(frame);
            std::string format = std::to_string(frame);
            for (int t = 0; t < 6 - unformat.length(); t++)
                format = "0" + format;
            time_stamps_.push_back(seq_name + "frame-" + format);
            depths_filenames_.push_back(dataset_dir + seq_name + "frame-" + format + ".depth.png");
            std::string gt_pose_name = dataset_dir + seq_name + "frame-" + format + ".pose.txt";
            Eigen::Matrix4f pose = loadTxtMatrix(gt_pose_name, 4, 4);
            gt_poses_.emplace_back(pose);
            ++count;
        }
    }
    size_ = count;
}

void seven_scenes_Dataset::readInfo(std::string &filename, std::vector<int> &start_frames,
                                    std::vector<int> &end_frames,
                                    std::vector<std::string> &seq_names)
{
    std::ifstream inFile;
    inFile.open(filename);
    Eigen::Matrix4f pose;
    pose.setIdentity();
    std::string line;
    int count = 0;
    while (getline(inFile, line))
    {
        std::string x;
        std::stringstream ss(line);
        while (ss >> x)
        {
            if (count == 0)
                start_frames.emplace_back(atoi(x.c_str()));
            if (count == 1)
                end_frames.emplace_back(atoi(x.c_str()));
            if (count == 2)
                seq_names.emplace_back("seq-" + x + "/");
        }
        count++;
    }
    inFile.close();
}

ICL_Dataset::ICL_Dataset(std::string &dataset_dir, int start_frame, int end_frame, bool is_flip)
{
    int count = 0;
    is_flip_ = is_flip;
    factor_ = 5;
    std::string poses_path = dataset_dir + "livingRoom1n.gt.sim";
    std::cout << "pose path:" << poses_path << std::endl;
    for (int i = start_frame; i <= end_frame; i++)
    {
        std::stringstream ss;
        std::string format = std::to_string(i);
        time_stamps_.emplace_back(format);
        depths_filenames_.emplace_back(dataset_dir + "depth/" + format + ".png");
        Eigen::Matrix4f gt_pose;
        readPoseFile(poses_path, 4 * i, 4 * i + 3, gt_pose);
        gt_poses_.emplace_back(gt_pose);
        count++;
    }
    size_ = count;
}

bool ICL_Dataset::readPoseFile(const std::string &poses_path, int start, int end, Eigen::Matrix4f &pose)
{
    std::ifstream poses_file(poses_path);
    if (!poses_file)
    {
        std::cout << "Error opening poses file." << std::endl;
        return false;
    }
    int i = 0;
    std::string temp;
    while (std::getline(poses_file, temp))
    {
        if (i < start)
            i++;
        else if (i >= start && i < end)
        {
            int j = 0;
            std::stringstream linestream(temp);
            std::string sub;
            while (linestream >> sub)
            {
                pose(i - start, j) = std::stod(sub);
                j++;
            }
            i++;
        }
        else
            break;
    }
    pose(3, 0) = 0.0;
    pose(3, 1) = 0.0;
    pose(3, 2) = 0.0;
    pose(3, 3) = 1.0;
    poses_file.close();
    return true;
}