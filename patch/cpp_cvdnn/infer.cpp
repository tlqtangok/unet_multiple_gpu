#include "infer.h"
#include <iostream>
#include <fstream>

UNetInference::UNetInference(const std::string& model_path, bool use_gpu, int batch_size)
    : _model_path(model_path), _use_gpu(use_gpu), _batch_size(batch_size), _model_loaded(false)
{
}

bool UNetInference::_validate_model_file() const
{
    std::ifstream file(_model_path);
    if (!file.good()) 
    {
        std::cerr << "Model file not found: " << _model_path << std::endl;
        return false;
    }
    return true;
}

void UNetInference::_setup_backend()
{
    if (_use_gpu) 
    {
        try 
        {
            _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using GPU backend" << std::endl;
        }
        catch (...) 
        {
            std::cout << "GPU failed, using CPU" << std::endl;
            _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }
    else 
    {
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "Using CPU backend" << std::endl;
    }
}

bool UNetInference::_test_forward()
{
    try 
    {
        cv::Mat test_input = cv::Mat::zeros(MODEL_SIZE, MODEL_SIZE, CV_8UC3);
        cv::Mat blob = _create_single_blob(test_input);
        
        _net.setInput(blob, "input");
        cv::Mat output = _net.forward("output");
        
        std::cout << "Model test successful" << std::endl;
        std::cout << "Output shape: [";
        for (int i = 0; i < output.dims; i++) 
        {
            std::cout << output.size[i];
            if (i < output.dims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        return true;
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "Model test failed: " << e.what() << std::endl;
        return false;
    }
}

bool UNetInference::load_model()
{
    if (!_validate_model_file()) 
    {
        return false;
    }

    try 
    {
        std::cout << "Loading model: " << _model_path << std::endl;
        _net = cv::dnn::readNetFromONNX(_model_path);
        
        if (_net.empty()) 
        {
            std::cerr << "Failed to load network" << std::endl;
            return false;
        }
        
        _setup_backend();
        
        if (!_test_forward()) 
        {
            return false;
        }
        
        _model_loaded = true;
        std::cout << "Model loaded successfully with batch size: " << _batch_size << std::endl;
        return true;
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "Load error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat UNetInference::_preprocess_image(const cv::Mat& input) const
{
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(MODEL_SIZE, MODEL_SIZE));
    return resized;
}

cv::Mat UNetInference::_create_single_blob(const cv::Mat& image) const
{
    return cv::dnn::blobFromImage(
        image, 
        1.0/255.0, 
        cv::Size(MODEL_SIZE, MODEL_SIZE), 
        cv::Scalar(), 
        true, 
        false
    );
}

void UNetInference::print_blob_detail(const cv::Mat& blob, const std::string& name)
{
	std::cout << name << " dimensions: ";

	if (blob.dims == 4)
	{
		std::cout << "[" << blob.size[0] << ", " << blob.size[1]
			<< ", " << blob.size[2] << ", " << blob.size[3] << "]" << std::endl;
		std::cout << "  [batch, channels, rows, cols] = ["
			<< blob.size[0] << "," << blob.size[1] << ","
			<< blob.size[2] << "," << blob.size[3] << "]" << std::endl;
	}
	else
	{
		std::cout << "[";
		for (int i = 0; i < blob.dims; i++)
		{
			std::cout << blob.size[i];
			if (i < blob.dims - 1)
			{
				std::cout << ", ";
			}
		}
		std::cout << "] (dims=" << blob.dims << ")" << std::endl;

		std::cout << "  [batch, channels, rows, cols] = [";
		for (int i = 0; i < blob.dims; i++)
		{
			std::cout << blob.size[i];
			if (i < blob.dims - 1)
			{
				std::cout << ",";
			}
		}
		std::cout << "]" << std::endl;
	}

	std::cout << "  type: " << blob.type() << std::endl;
	std::cout << "  total elements: " << blob.total() << std::endl;
}
cv::Mat UNetInference::_create_batch_blob(const std::vector<cv::Mat>& images) const
{
    return cv::dnn::blobFromImages(
        images, 
        1.0/255.0, 
        cv::Size(MODEL_SIZE, MODEL_SIZE), 
        cv::Scalar(), 
        true, 
        false
    );
}

cv::Mat UNetInference::_extract_mask_single(const cv::Mat& output) const
{
    if (output.dims != 4 || output.size[1] != 2) 
    {
        std::cerr << "Invalid output format" << std::endl;
        return cv::Mat();
    }
    
    int height = output.size[2];
    int width = output.size[3];
    
    float* data = (float*)output.data;
    
    cv::Mat class0(height, width, CV_32F);
    cv::Mat class1(height, width, CV_32F);
    
    int channel_size = height * width;
    
    for (int i = 0; i < channel_size; i++) 
    {
        class0.at<float>(i / width, i % width) = data[i];
        class1.at<float>(i / width, i % width) = data[i + channel_size];
    }
    
    cv::Mat mask;
    cv::compare(class1, class0, mask, cv::CMP_GT);
    
    return mask;
}

std::vector<cv::Mat> UNetInference::_extract_masks_batch(const cv::Mat& output) const
{
    std::vector<cv::Mat> masks;
    
    if (output.dims != 4) 
    {
        std::cerr << "Invalid batch output dimensions: " << output.dims << std::endl;
        return masks;
    }
    
    int batch_size = output.size[0];
    int num_classes = output.size[1];
    int height = output.size[2];
    int width = output.size[3];
    int image_size = num_classes * height * width;
    
    if (num_classes != 2) 
    {
        std::cerr << "Invalid batch output channels: " << num_classes << std::endl;
        return masks;
    }
    
    float* data = (float*)output.data;
    int channel_size = height * width;
    
    for (int b = 0; b < batch_size; b++) 
    {
        cv::Mat class0(height, width, CV_32F);
        cv::Mat class1(height, width, CV_32F);
        
        int batch_offset = b * image_size;
        
        // Use memcpy for channel 0 (background)
        memcpy(class0.data, 
               data + batch_offset, 
               channel_size * sizeof(float));
        
        // Use memcpy for channel 1 (foreground)  
        memcpy(class1.data, 
               data + batch_offset + channel_size, 
               channel_size * sizeof(float));
        
        cv::Mat mask;
        cv::compare(class1, class0, mask, cv::CMP_GT);
        masks.push_back(mask);
    }
    
    return masks;
}


cv::Mat UNetInference::_resize_to_original(const cv::Mat& mask, const cv::Size& original_size) const
{
    cv::Mat resized;
    cv::resize(mask, resized, original_size, 0, 0, cv::INTER_NEAREST);
    return resized;
}

cv::Mat UNetInference::predict_single(const cv::Mat& image)
{
    if (!_model_loaded) 
    {
        std::cerr << "Model not loaded" << std::endl;
        return cv::Mat();
    }

    try 
    {
        cv::Size original_size = image.size();
        
        cv::Mat processed = _preprocess_image(image);
        cv::Mat blob = _create_single_blob(processed);
        
        _net.setInput(blob, "input");
        cv::Mat output = _net.forward("output");
        
        cv::Mat mask = _extract_mask_single(output);
        if (mask.empty()) 
        {
            return cv::Mat();
        }
        
        cv::Mat result = _resize_to_original(mask, original_size);
        
        return result;
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "Prediction error: " << e.what() << std::endl;
        return cv::Mat();
    }
}

std::vector<cv::Mat> UNetInference::predict_batch(const std::vector<cv::Mat>& images)
{
    std::vector<cv::Mat> results;
    
    if (!_model_loaded) 
    {
        std::cerr << "Model not loaded" << std::endl;
        return results;
    }
    
    if (images.empty()) 
    {
        return results;
    }

    try 
    {
        std::vector<cv::Size> original_sizes;
        std::vector<cv::Mat> processed_images;
        
        for (const auto& image : images) 
        {
            original_sizes.push_back(image.size());
            processed_images.push_back(_preprocess_image(image));
        }
        
        cv::Mat blob = _create_batch_blob(processed_images);

		std::cout << "blob.size():" << blob.size() << std::endl; 


		print_blob_detail(blob, "input batch"); 

        
        _net.setInput(blob, "input");
        cv::Mat output = _net.forward("output");
		print_blob_detail(output, "output batch"); 


        
        std::vector<cv::Mat> masks = _extract_masks_batch(output);
        
        if (masks.size() != original_sizes.size()) 
        {
            std::cerr << "Batch size mismatch" << std::endl;
            return results;
        }
        
        for (size_t i = 0; i < masks.size(); i++) 
        {
            cv::Mat result = _resize_to_original(masks[i], original_sizes[i]);
            results.push_back(result);
        }
        
        return results;
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "Batch prediction error: " << e.what() << std::endl;
        return results;
    }
}

bool UNetInference::is_model_loaded() const
{
    return _model_loaded;
}

int UNetInference::get_batch_size() const
{
    return _batch_size;
}

