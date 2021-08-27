# gan_hls

#### Youtube Video
[![Youtube video](/docs/cover.png)](https://www.youtube.com/watch?v=FO_M2AHb1u4)

gan-hls is a Deep Learning project that implements an Image Reconstruction algorithm with GANs (Generative Adversarial Networks). The Generator neural network model trained on Tensorflow takes as input the top half of a clothing image and generates/predicts the bottom half. It was then implemented in C++ and was accelerated with Xilinx Vitis using High Level Synthesis to run on cloud FPGAs (such as Xilinx Alveos). The architecure also achieves significant performance and power efficiency and different fixed point precisions can be tested (8-bit, 6-bit, etc.)


### Features
- Application: Accelerated Image reconstruction with GANs and cloud FPGAs
- Input data: 28x28 greyscale images
- Dataset:  Fashion MNIST clothing images (60K images train set, 10K images test set)
- Classes: 10 	
- PSNR: 48 dB (8-bit FPGA vs CPU)
- Inference Speed (FPGA): 50ms (on whole dataset)


---

## System Setup

- Ubuntu Linux (18.04 recommended)
- Vitis 2020.2 installed with XRT 2.9.317
- Alveo U50 FPGA device installed and configured
- (Optional) Jupyter Notebook for visualizing output


## HLS compilation

The project can be compiled for a range of devices. However, it is tested for DEVICE=xilinx_u50_gen3x16_xdma_201920_3.

- You can run the following command to emulate in SW the application:
```bash
make all TARGET=sw_emu DEVICE=xilinx_u50_gen3x16_xdma_201920_3
``` 
- Or you run ran the following command to compile the application and generate the bitstream of the FPGA:
```bash  
make all TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3
``` 
- If you want to compile only the host code you can simply run:
```bash
make host TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3
``` 
For emulation modes only you must set the XCL_EMULATION_MODE environment variable:
```bash
export XCL_EMULATION_MODE=<sw_emu,hw_emu> 
``` 
Then to execute application with the generated bitstream:
```bash
./application_gan <path_to_bitstream>
``` 

- (Optional): The project uses 8-bit quantization by default. If you would like to use different precision simply change the BITS and BITS_EXP in [network.h](src/network.h)  according to the comments in the file. Also replace the contents of [tanh.h](src/tanh.h) with the appropriate pre-computed values of Tanh function of the precision you selected (.i.e. for 6-bits you must replace it with the contents of tanh_6.h). Then re-build the whole project.

## Demo

Execute the following commands to compile the host code and run the application with the pre-synthesized bitstream.

```bash
make host TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3
./application_gan network.xclbin
``` 

The application produces an "output.txt" which contains the generated half images of the input dataset (one by one the pixels). In order to visualize the resuls run the jupyter notebook [plot_output.ipynb](plot_output.ipynb) which is provided.

#### Publication

In case you use some of the code please cite the following paper:

```
@misc{
  author =       "Dimitrios Danopoulos, Konstantinos Anagnostopoulos, Christoforos Kachris, and Dimitrios Soudris",
  title =        "FPGA Acceleration of Generative Adversarial Networks for Image Reconstruction",
  conference =   "The International Conference on Modern Circuits and Systems Technologies (MOCAST)",
  year =         "2021",
  month =        "July",
}
```
#### Acknowledgements

This project has received funding from the Hellenic Foundation for Research and Innovation (HFRI) and the General Secretariat for Research and Technology (GSRT) under grant agreement no 2212 and the Xilinx University Program.
    <img src="http://www.elidek.gr/wp-content/uploads/2018/05/HFRI_LOGO_SMALL.png" alt="" width=355 height=164>
    <img src="https://www.cl.cam.ac.uk/teaching/1011/P33/logo/xup.jpg" alt="" width=125 height=126>

## Contact

Dimitrios Danopoulos: dimdano@microlab.ntua.gr
    
