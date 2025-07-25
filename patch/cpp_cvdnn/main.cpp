#include "infer.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

struct Config 
{
    std::string model_path;
    std::string input_path;
    std::string output_path;
    bool use_gpu = false;
    int batch_size = 1;
};

bool parse_arguments(int argc, char* argv[], Config& config)
{
    for (int i = 1; i < argc; i++) 
    {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) 
        {
            config.model_path = argv[++i];
        }
        else if (arg == "--input" && i + 1 < argc) 
        {
            config.input_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) 
        {
            config.output_path = argv[++i];
        }
        else if (arg == "--gpu") 
        {
            config.use_gpu = true;
        }
        else if (arg == "--cpu") 
        {
            config.use_gpu = false;
        }
        else if (arg == "--batch" && i + 1 < argc) 
        {
            config.batch_size = std::stoi(argv[++i]);
        }
    }
    
    return !config.model_path.empty() && !config.input_path.empty() && !config.output_path.empty();
}

void print_config(const Config& config)
{
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model: " << config.model_path << std::endl;
    std::cout << "  Input: " << config.input_path << std::endl;
    std::cout << "  Output: " << config.output_path << std::endl;
    std::cout << "  Device: " << (config.use_gpu ? "GPU" : "CPU") << std::endl;
    std::cout << "  Batch: " << config.batch_size << std::endl;
}

std::vector<std::string> get_image_files(const std::string& input_path)
{
    std::vector<std::string> files;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    
    for (const auto& entry : std::filesystem::directory_iterator(input_path)) 
    {
        if (entry.is_regular_file()) 
        {
            std::string path = entry.path().string();
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) 
            {
                if (path.find("_OUT.png") == std::string::npos)
                {
                    files.push_back(path);
                }
            }
        }
    }
    
    std::sort(files.begin(), files.end());
    return files;
}

bool create_output_directory(const std::string& output_path)
{
    try 
    {
        std::filesystem::create_directories(output_path);
        return true;
    }
    catch (...) 
    {
        std::cerr << "Failed to create output directory: " << output_path << std::endl;
        return false;
    }
}

std::string extract_filename_without_extension(const std::string& filepath)
{
    std::filesystem::path path(filepath);
    return path.stem().string();
}

std::string extract_directory(const std::string& filepath)
{
    std::filesystem::path path(filepath);
    return path.parent_path().string();
}

std::string generate_output_filename_from_original(const std::string& original_filepath, const std::string& output_dir)
{
    std::string base_name = extract_filename_without_extension(original_filepath);
    std::string output_filename = base_name + "_OUT.png";
    
    if (output_dir.empty())
    {
        std::string original_dir = extract_directory(original_filepath);
        return std::filesystem::path(original_dir) / output_filename;
    }
    
    return std::filesystem::path(output_dir) / output_filename;
}

std::vector<cv::Mat> load_batch_images(const std::vector<std::string>& image_files, size_t start_idx, int batch_size)
{
    std::vector<cv::Mat> batch_images;
    
    for (int i = 0; i < batch_size && (start_idx + i) < image_files.size(); i++) 
    {
        cv::Mat image = cv::imread(image_files[start_idx + i]);
        if (!image.empty()) 
        {
            batch_images.push_back(image);
        }
        else 
        {
            std::cerr << "Failed to load image: " << image_files[start_idx + i] << std::endl;
        }
    }
    
    return batch_images;
}

bool save_batch_results(const std::vector<cv::Mat>& results, const std::vector<std::string>& image_files, size_t start_idx, const std::string& output_dir)
{
    for (size_t i = 0; i < results.size(); i++) 
    {
        std::string output_file = generate_output_filename_from_original(image_files[start_idx + i], output_dir);
        
        if (!cv::imwrite(output_file, results[i])) 
        {
            std::cerr << "Failed to save: " << output_file << std::endl;
            return false;
        }
        
        std::cout << "Saved: " << output_file << std::endl;
    }
    
    return true;
}

bool process_images_with_batch(UNetInference& inference, const std::vector<std::string>& image_files, const std::string& output_dir)
{
    int batch_size = inference.get_batch_size();
    
    std::cout << "Processing " << image_files.size() << " images with batch size " << batch_size << std::endl;
    
    for (size_t i = 0; i < image_files.size(); i += batch_size) 
    {
        std::cout << "Processing batch " << (i / batch_size + 1) << "/" << ((image_files.size() + batch_size - 1) / batch_size) << std::endl;
        
        std::vector<cv::Mat> batch_images = load_batch_images(image_files, i, batch_size);
        
        if (batch_images.empty()) 
        {
            continue;
        }
        
        std::vector<cv::Mat> results;
        
        if (batch_images.size() == 1) 
        {
            cv::Mat single_result = inference.predict_single(batch_images[0]);
            if (!single_result.empty()) 
            {
                results.push_back(single_result);
            }
        }
        else 
        {
            results = inference.predict_batch(batch_images);
        }
        
        if (results.empty()) 
        {
            std::cerr << "Failed to predict batch starting at index " << i << std::endl;
            continue;
        }
        
        if (!save_batch_results(results, image_files, i, output_dir)) 
        {
            return false;
        }
    }
    
    return true;
}

// main_
int main(int argc, char* argv[])
{
    Config config;
    
    if (!parse_arguments(argc, argv, config)) 
    {
        std::cerr << "Usage: " << argv[0] << " --model <model.onnx> --input <input_dir> --output <output_dir> [--gpu|--cpu] [--batch <size>]" << std::endl;
        return 1;
    }
    
    print_config(config);
    
    UNetInference inference(config.model_path, config.use_gpu, config.batch_size);
    
    if (!inference.load_model()) 
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    if (!create_output_directory(config.output_path)) 
    {
        return 1;
    }
    
    std::vector<std::string> image_files = get_image_files(config.input_path);
    if (image_files.empty()) 
    {
        std::cerr << "No images found in: " << config.input_path << std::endl;
        return 1;
    }
    
    std::cout << "Found " << image_files.size() << " images" << std::endl;
    
    if (!process_images_with_batch(inference, image_files, config.output_path)) 
    {
        return 1;
    }
    
    std::cout << "Processing completed" << std::endl;
    return 0;
}

