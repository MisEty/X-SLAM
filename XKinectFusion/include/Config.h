#ifndef GRAD_KINECTFUSION_IO_HELPER_HPP
#define GRAD_KINECTFUSION_IO_HELPER_HPP
#include <opencv2/core.hpp>

#include <iostream>
#include <memory>

class Config {
public:
    std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() = default;

    ~Config() {
        if (file_.isOpened()) {
            file_.release();
        }
    }

    bool read(const std::string& filename) {
        if (config_ == nullptr) {
            config_ = std::make_shared<Config>();
        }

        config_->file_.open(filename.c_str(), cv::FileStorage::READ);
        if (!config_->file_.isOpened()) {
            std::cout << filename << " does not exist!!!\n";
            config_->file_.release();
            return false;
        }
        return true;
    }
};

template<typename T>
inline static T get(const Config& config, const std::string& key) {
    return T(config.config_->file_[key]);
}
#endif//GRAD_KINECTFUSION_IO_HELPER_HPP