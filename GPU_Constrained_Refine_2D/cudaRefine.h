#ifndef REFINE_H
#define REFINE_H

#include "cudaThrust.h"

void GPU_Refine_Quality(triangulateio *input, triangulateio *result, double input_theta, double input_size, InsertPolicy insertpolicy,DeletePolicy deletepolicy, int encmode, int runmode,
	int filtermode, int unifymode, int debug_iter, PStatus **ps_debug,TStatus **ts_debug);

#endif