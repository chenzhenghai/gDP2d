# gDP2d
2D Constrained Delaunay Refinement on the GPU

Project Website: https://www.comp.nus.edu.sg/~tants/gqm.html

Paper:  
(1) Computing Delaunay Refinement Using the GPU. Z. Chen, M. Qi, and T.S. Tan. The 2017 ACM Symposium on Interactive 3D Graphics and Games, 25-27 Feb, San Francisco, CA, USA. (<a href="https://www.comp.nus.edu.sg/~tants/gqm_files/11-0018-chen.pdf">PDF</a>)  
(2) On Designing GPU Algorithms with Applications to Mesh Refinement. Z. Chen, T.S. Tan, and H.Y. Ong. arXiv, 2020. (<a href="https://arxiv.org/abs/2007.00324">PDF</a>)


* A NVIDIA GPU is required since this project is implemented using CUDA  
* The development environment: Visual Studio 2017 and CUDA 9.0 (Please use x64 and Release mode.)

--------------------------------------------------------------------------
Refinement Routine (located in refine.h and refine.cu):  
void GPU_Refine_Quality(  
&nbsp;&nbsp;&nbsp;&nbsp; triangulateio *input,  
&nbsp;&nbsp;&nbsp;&nbsp; triangulateio *result,  
&nbsp;&nbsp;&nbsp;&nbsp; double theta,  
&nbsp;&nbsp;&nbsp;&nbsp; InsertPolicy insertpolicy,  
&nbsp;&nbsp;&nbsp;&nbsp; DeletePolicy deletepolicy,  
&nbsp;&nbsp;&nbsp;&nbsp; int mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int debug_iter,  
&nbsp;&nbsp;&nbsp;&nbsp; PStatus **ps_debug,  
&nbsp;&nbsp;&nbsp;&nbsp; TStatus **ts_debug)  