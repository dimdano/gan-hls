#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include "network.h"
#include "software_weight_definitions.h"
#include <math.h>
#include <sys/time.h>
#include "xcl2.hpp"
#include <chrono>

// select number of images to process from dataset (max 10K)
//
#define INPUT_IMAGES 10000
auto constexpr num_cu = 4;

static inline float ReLu(float value)
{
	if (value < 0)
		return 0;
	return value;
}

//the Generator model implemented in software
void sw_forward_propagation(std::vector<float, aligned_allocator<float> > & input, std::vector<float, aligned_allocator<float> > & output, int images)
{
	float layer_1_out[M1];
	float layer_2_out[M2];

	for(int iter = 0; iter < images; ++iter){
	// Layer 1
	for (int i=0; i<M1; i++)
	{
		float result = 0;
		for (int j=0; j<N1; j++)
		{
			float term = input[iter*M3+j] * W1_sw[i][j];
			result += term;
		}
		layer_1_out[i] = ReLu(result);
	}

	// Layer 2
	for (int i=0; i<M2; i++)
	{
		float result = 0;
		for (int j=0; j<N2; j++)
		{
			float term = layer_1_out[j] * W2_sw[i][j];
			result += term;
		}
		layer_2_out[i] = ReLu(result);
	}

	// Layer 3
	for (int i=0; i<M3; i++)
	{
		float result = 0;
		for (int j=0; j<N3; j++)
		{
			float term = layer_2_out[j] * W3_sw[i][j];
			result += term;
		}
		output[iter*M3+i] = tanh(result);
	}
	}
}

void flush_array(char *ar, int size)
{
    for(int i=0; i<size; i++)
    {
        ar[i] = '\0';
    }
}

void copy_ar(float *source, float *dest)
{
	for (int i=0; i<392; i++)
		dest[i] = source[i];
}

void parse_dataset(std::vector<float, aligned_allocator<float> > & input, int batch_number)
{
	 /* Parse CSV file containing in each row pixel values for half
	 cut images from the whole test set */

	FILE *fp;
	fp = fopen("data.txt", "r");

	int c;
	char number[20] = {'\0'};
	int i = 0;
	int I = 0;

	do
	{
		c = fgetc(fp);
		if (c != ';' && c != '\n')
		{
			number[i] = c;
			i++;
		}
		else
		{
			float value = atof(number);
			input[I] = value;
			I++;
			i=0;
			flush_array(number,20);
		}

	}while ( c != EOF && I<batch_number*392);

	fclose(fp);
}


using namespace std;

int main(int argc, char* argv[])
{
	
	//check for xclbin input
	if (argc != 2) {
        	std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        	return EXIT_FAILURE;
	}

	std::cout << endl;
	std::cout << "|******************************************************|" << endl;
    std::cout << "|                                                      |" << endl;
    std::cout << "|                        GAN-HLS                       |" << endl;
    std::cout << "| Accelerated image reconstruction with GANs and FPGAs |" << endl;
    std::cout << "|                                                      |" << endl;
    std::cout << "|******************************************************|" << endl;


	std::string binaryFile = argv[1];
	cl_int err;
	cl::Context context;
	cl::Kernel network;
	cl::CommandQueue q;


    cout << "FPGA configuration started." << endl;

	auto devices = xcl::get_xil_devices();
	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	
	for (unsigned int i = 0; i < devices.size(); i++) {
    	auto device = devices[i];
    	// Creating Context and Command Queue for selected Device
    	context = cl::Context(device, nullptr, nullptr, nullptr, &err);
    	q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    	std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    	cl::Program program(context, {device}, bins, nullptr, &err);
    	if (err != CL_SUCCESS) {
        		std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    	} else {
        		std::cout << "Device[" << i << "]: program successful!\n";
        		network = cl::Kernel(program, "network", &err);
        		valid_device = true;
        		break; // we break because we found a valid device
    	}
	}
	if (!valid_device) {
    	std::cout << "Failed to program any device found, exit!\n";
    	exit(EXIT_FAILURE);
	}

	cout << "--------------------------------" << endl;

	//std::vector<cl::Kernel> krnls(num_cu);

	//allocate memory alligned vectors for input and output
    std::vector<float, aligned_allocator<float> > source_in(N1 * INPUT_IMAGES * sizeof(float));
    std::vector<float, aligned_allocator<float> > source_out(M3 * INPUT_IMAGES * sizeof(float));
	std::vector<float, aligned_allocator<float> > source_swout(M3 * INPUT_IMAGES * sizeof(float));

	cout << "Starting dataset parsing..." << endl;
	parse_dataset(source_in, INPUT_IMAGES);
	cout << "Parsing finished." << endl;
    cout << "--------------------------------" << endl;

	//allocate OpenCL buffers
	cl::Buffer buffer_in(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N1 * INPUT_IMAGES * sizeof(float),
                                        source_in.data(), &err);
	cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, M3 * INPUT_IMAGES * sizeof(float),
                                        source_out.data(), &err);
   
	//specify kernel arguments
	err = network.setArg(0, buffer_in);
  	err = network.setArg(1, buffer_out);
	err = network.setArg(2, INPUT_IMAGES);

	//transfer data from host to global memory
	err = q.enqueueMigrateMemObjects({buffer_in}, 0 /* 0 means from host*/);	
	
	//initiate kernel
	cout << "Starting hardware calculations..." << endl;
	auto ts = std::chrono::high_resolution_clock::now();
	//execute kernel in HW
	err = q.enqueueTask(network);
	q.finish();
	auto te = std::chrono::high_resolution_clock::now();
	cout << "Ended hardware calculations." << endl;
	float timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count() / 1000.0f;
	std::cout << "FPGA time: " << timeTaken << "ms" << endl;
       
	//transfer data from global memory to host
	cout << "Sending results to host..." << endl;
    	err = q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST);
        cout << "Results sent." << endl;
    	cout << "--------------------------------" << endl;
	err = q.finish();

    
	//calculate golden output using float32 datatype
	cout << "Starting software calculations..." << endl;

	auto ss = std::chrono::high_resolution_clock::now();
	//execute kernel in SW
	sw_forward_propagation(source_in,source_swout,INPUT_IMAGES);
	auto se = std::chrono::high_resolution_clock::now();	
	float timeTaken_sw = std::chrono::duration_cast<std::chrono::microseconds>(se - ss).count() / 1000.0f;
	cout << "Ended software calculations." << endl;
        std::cout << "CPU time: " << timeTaken_sw << "ms" << endl;
	
	
	cout << "--------------------------------" << endl;
	
	//save output of generated images in file
	FILE *fp;
	fp = fopen("output.txt", "w");
	fprintf(fp, "Software Results;Hardware Results\n");
	cout << "Saving results to output.txt..." << endl;

	for(int i=0 ; i<INPUT_IMAGES * M3; i++)
	{
		fprintf(fp, "%f;%f\n",source_swout[i], source_out[i]);
	}
	fclose(fp);
	
	cout << "Results written to file." << endl;
    cout << "You can plot results with the jupyter notebook provided." << endl;

	return 0;
}
