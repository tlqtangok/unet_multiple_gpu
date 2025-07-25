#ifndef UNET_INFERENCE_H
#define UNET_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class UNetInference 
{
public:
    UNetInference(const std::string& model_path, bool use_gpu = false, int batch_size = 1);
    
    bool load_model();
    cv::Mat predict_single(const cv::Mat& image);
    std::vector<cv::Mat> predict_batch(const std::vector<cv::Mat>& images);
    bool is_model_loaded() const;
    int get_batch_size() const;
    
private:
    bool _validate_model_file() const;
    void _setup_backend();
    bool _test_forward();
    
    cv::Mat _preprocess_image(const cv::Mat& input) const;
    cv::Mat _create_single_blob(const cv::Mat& image) const;
    cv::Mat _create_batch_blob(const std::vector<cv::Mat>& images) const;
	void print_blob_detail(const cv::Mat& blob, const std::string& name);
    cv::Mat _extract_mask_single(const cv::Mat& output) const;
    std::vector<cv::Mat> _extract_masks_batch(const cv::Mat& output) const;
    cv::Mat _resize_to_original(const cv::Mat& mask, const cv::Size& original_size) const;
    
    std::string _model_path;
    bool _use_gpu;
    int _batch_size;
    bool _model_loaded;
    cv::dnn::Net _net;
    
    static const int MODEL_SIZE = 160;
};

#endif

