# gDP2d
2D Constrained Delaunay Refinement on the GPU

Project Website: https://www.comp.nus.edu.sg/~tants/gqm.html

Paper:  
1. Computing Delaunay Refinement Using the GPU. Z. Chen, M. Qi, and T.S. Tan. The 2017 ACM Symposium on Interactive 3D Graphics and Games, 25-27 Feb, San Francisco, CA, USA. (<a href="https://www.comp.nus.edu.sg/~tants/gqm_files/11-0018-chen.pdf">PDF</a>)  
2. On Designing GPU Algorithms with Applications to Mesh Refinement. Z. Chen, T.S. Tan, and H.Y. Ong. arXiv, 2020. (<a href="https://arxiv.org/abs/2007.00324">PDF</a>)


* A NVIDIA GPU is required since this project is implemented using CUDA  
* The development environment: Visual Studio 2017 and CUDA 9.0 (Please use x64 and Release mode.)

--------------------------------------------------------------------------
Refinement Routine for synthetic and real-world datasets (located in GPU_Constrained_Refine_2D/meshRefine.h and meshRefineGPU.cpp):

void refineInputByGPU_Synthetic(  
&nbsp;&nbsp;&nbsp;&nbsp; char* input_path,  
&nbsp;&nbsp;&nbsp;&nbsp; double input_theta,  
&nbsp;&nbsp;&nbsp;&nbsp; double input_size,  
&nbsp;&nbsp;&nbsp;&nbsp; int enc_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int run_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int filter_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int unify_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; SyntheticParam input_sparam,  
&nbsp;&nbsp;&nbsp;&nbsp; char* output_path)  

void refineInputByGPU_Real(  
&nbsp;&nbsp;&nbsp;&nbsp; char* input_path,  
&nbsp;&nbsp;&nbsp;&nbsp; char* input_file,  
&nbsp;&nbsp;&nbsp;&nbsp; double input_theta,  
&nbsp;&nbsp;&nbsp;&nbsp; double input_size,  
&nbsp;&nbsp;&nbsp;&nbsp; int enc_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int run_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int filter_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int unify_mode,  
&nbsp;&nbsp;&nbsp;&nbsp; char* output_path)  

char* input_path:  
The path for the folder that contains input files.

char* input_file:  
Input filename.

double input_theta:  
Minimum allowable angle. Theoretically, it cannot be smaller than 20.7 degree. The triangle in final mesh wouldn't contain angles smaller than input_theta, except those exist in the input PSLG.

double input_size:  
Maximum allowable edge length. The final mesh wouldn't contain edges longer than input_size.

int enc_mode:  
Encroachment mode. When mode is 1, Ruppert's algorithm is used; otherwise, Chew's is used.

int run_mode:  
Running mode. When mode is 1, the new approach in Paper 2 is used; otherwise, the old approach in Paper 1 is used.

int filter_mode:  
Filtering mode. When mode is 1, fast filtering is used; otherwise, no filtering is used.

int unify_mode:  
Unifying mode. When mode is 1, subsegments and triangles are split together; otherwise, they are split separately.

SyntheticParam input_sparam:  
The input parameters for synthetic datasets; see GPU_Constrained_Refine_2D/meshRefine.h

char* output_path:  
The path for the folder to contain output files.

--------------------------------------------------------------------------
Proceed to GPU_Constrained_Refine_2D/main.cpp to check how to call GPU refinement routines properly.