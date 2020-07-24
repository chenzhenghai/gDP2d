#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaThrust.h"

//#define GQM2D_DEBUG
//#define GQM2D_DEBUG_2
//#define GQM2D_PROFILING
//#define GQM2D_LOOP_PROFILING
//#define GQM2D_CHECKMEMORY

#define GQM2D_INEXACT
#define GQM2D_PRIORITY_SIZE // comment out this for synthetic datasets

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void gpuMemoryCheck();

void cudaexactinit();

#define CAVITY_FAST_SIZE 50 // >= 10

// Flip flop kernels
__global__ void kernelInitFlipOri
(
	TStatus	  *d_TStatus,
	int	numberoftriangle
);

__global__ void kernelCheckFlipping
(
	REAL2	  *d_pointlist,
	PStatus	  *d_PStatus,
	int		  *d_trianglelist,
	int		  *d_neighborlist,
	int		  *d_tri2subseg,
	TStatus   *d_TStatus,
	int		  *d_flipBy,
	int		  *d_active,
	int		  numberofactive
);

__global__ void kernelCheckFlipping_New
(
	REAL2	  *d_pointlist,
	PStatus	  *d_PStatus,
	REAL	  *d_pointpriority,
	int		  *d_trianglelist,
	int		  *d_neighborlist,
	int		  *d_tri2subseg,
	TStatus   *d_TStatus,
	int		  *d_flipBy,
	int		  *d_active,
	int		  numberofactive
);

__global__ void kernelMarkFlipWinner
(
	PStatus 	*d_PStatus,
	int		*d_trianglelist,
	int        *d_neighborlist,
	TStatus	*d_TStatus,
	int		*d_flipBy,
	int		*d_active,
	int		numberofactive
);

__global__ void kernelInitLinklist(
	PStatus    *d_PStatus,
	int		*d_trianglelist,
	int        *d_neighborlist,
	TStatus	*d_TStatus,
	int		*d_active,
	int		*d_linklist,
	int		numberofactive,
	int * d_tri2subseg,
	int iteration0,
	int iteration1,
	int step
);

__global__ void kernelUpdatePhase1
(
	PStatus 	*d_PStatus,
	int		*d_trianglelist,
	int        *d_neighborlist,
	int		*d_tri2subseg,
	TStatus	*d_TStatus,
	int		*d_subseg2tri,
	int		*d_flipBy,
	int		*d_active,
	int		*d_linkslot,
	int		numberofactive,
	int iteration0,
	int iteration1,
	int step
);

__global__ void kernelUpdatePhase2
(
	PStatus 	*d_PStatus,
	int		*d_trianglelist,
	int        *d_neighborlist,
	int		*d_tri2subseg,
	TStatus	*d_TStatus,
	int		*d_subseg2tri,
	int		*d_flipBy,
	int		*d_active,
	int		*d_linklist,
	int		*d_linkslot,
	int		numberofactive,
	int iteration
);

__global__ void kernelUpdatePhase3
(
	REAL2		*d_pointlist,
	PStatus 	*d_PStatus,
	int		*d_trianglelist,
	int        *d_neighborlist,
	int		*d_tri2subseg,
	TStatus	*d_TStatus,
	int		*d_subseg2tri,
	int		*d_flipBy,
	int		*d_active,
	int		*d_encmarker,
	int		*d_linklist,
	int		*d_linkslot,
	int		numberofactive,
	int run_mode,
	REAL		theta,
	int iteration,
	int step,
	int last_triangle
);

__global__ void kernelUpdatePhase3_New
(
	REAL2   *d_pointlist,
	PStatus *d_PStatus,
	int		*d_trianglelist,
	int     *d_neighborlist,
	int		*d_tri2subseg,
	TStatus	*d_TStatus,
	int		*d_subseg2tri,
	int		*d_flipBy,
	int		*d_active,
	int		*d_encmarker,
	int     *d_segmarker,
	int		*d_linklist,
	int		*d_linkslot,
	int		numberofactive,
	int encmode,
	REAL theta
);

__global__ void kernelMarkDiametralCircle
(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_trianglelist,
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_active,
	int numberofactive
);

__global__ void kernelMarkReduntantSteiner
(
	PStatus 	*d_PStatus,
	int		*d_trianglelist,
	int        *d_neighborlist,
	TStatus	*d_TStatus,
	int		*d_flipBy,
	int		*d_active,
	int		numberofactive
);

// Split encroached subsegments

void markReduntantPoints(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	TStatusD &t_TStatus,
	IntD	 &t_subseg2tri,
	IntD	 &t_delmarker,
	IntD	 &t_dellist,
	int	numberofdels
);

__global__ void kernelResetMidInsertionMarker(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_trimarker,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive
);

__global__ void kernelMarkMidInsertion(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_trimarker,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive,
	int * d_trianglelist,
	int * d_tri2subseg,
	int iteration
);

__global__ void kernelInsertMidPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int	* d_trianglelist,
	int * d_neighborlist,
	int	* d_tri2subseg,
	TStatus * d_TStatus,
	int * d_segmentlist,
	int * d_subseg2tri,
	int * d_subseg2seg,
	int * d_encmarker,
	int * d_internalmarker,
	int * d_internallist,
	int * d_trimarker,
	int * d_emptypoints,
	int * d_emptytriangles,
	int emptypointsLength,
	int emptytrianglesLength,
	int numberofemptypoints,
	int numberofemptytriangles,
	int numberofsubseg,
	int numberofactive,
	int run_mode,
	REAL theta,
	int iteration
);

__global__ void kernelUpdateMidNeighbors
(
	REAL2 * d_pointlist,
	int	* d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_enclist,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive,
	int run_mode,
	REAL theta
);

// Split bad triangles

int updateActiveListToBadTriangles
(
	Real2D		&t_pointlist,
	PStatusD	&t_PStatus,
	IntD		&t_trianglelist,
	IntD		&t_neighborlist,
	IntD		&t_segmentlist,
	IntD		&t_subseg2seg,
	IntD		&t_tri2subseg,
	TStatusD	&t_TStatus,
	IntD		&t_active0,
	IntD		&t_active1,
	int         numberoftriangles,
	REAL theta,
	REAL size
);

__global__ void kernelComputeCircumcenter(
	REAL2 * d_pointlist,
	int * d_trianglelist,
	int * d_active,
	int numberofactive,
	REAL2 * d_TCenter,
	int * d_Priority,
	REAL offConstant,
	bool offCenter
);

__global__ void kernelLocateSinkPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_TCenter,
#ifdef GQM2D_PRIORITY_SIZE
	uint64* d_trimarker,
#else
	int * d_trimarker,
#endif
	int * d_sinks,
	int * d_Priority,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive
);

__global__ void kernelRemoveLosers
(
	int		  *d_trianglelist,
	int          *d_neighborlist,
	TStatus      *d_TStatus,
#ifdef GQM2D_PRIORITY_SIZE
	uint64* d_trimarker,
#else
	int * d_trimarker,
#endif
	int		  *d_sinks,
	int		  *d_Priority,
	int          *d_internalmarker,
	int		  *d_internallist,
	int		  numberofactive
);

void markCavities(
	Real2D &t_pointlist,
	IntD &t_trianglelist,
	IntD &t_neighborlist,
	IntD &t_tri2subseg,
	TStatusD &t_TStatus,
	Real2D &t_TCenter,
	IntD &t_sinks,
	IntD &t_Priority,
#ifdef GQM2D_PRIORITY_SIZE
	UInt64D &t_trimarker,
#else
	IntD &t_trimarker,
#endif
	IntD &t_internalmarker,
	IntD &t_internallist,
	int numberofbad,
	int maxnumofblock // for number of triangles
);

void checkCavities(
	Real2D &t_pointlist,
	IntD &t_trianglelist,
	IntD &t_neighborlist,
	IntD &t_tri2subseg,
	TStatusD &t_TStatus,
	Real2D &t_TCenter,
	IntD &t_sinks,
	IntD &t_Priority,
#ifdef GQM2D_PRIORITY_SIZE
	UInt64D &t_trimarker,
#else
	IntD &t_trimarker,
#endif
	IntD &t_internalmarker,
	IntD &t_internallist,
	int numberofbad,
	int maxnumofblock // for number of triangles
);

__global__ void kernelMarkSteinerOnsegs
(
	int * d_tri2subseg,
	int * d_enclist,
	int * d_internallist,
	int * d_sinks,
	int numberofactive
);

__global__ void kernelResetSteinerInsertionMarker(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_internallist,
	int * d_sinks,
	int numberofactive
);

__global__ void kernelInsertSteinerPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int	* d_trianglelist,
	int * d_neighborlist,
	int	* d_tri2subseg,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_internalmarker,
	int * d_internallist,
	REAL2 * d_TCenter,
	int	* d_sinks,
	int * d_emptypoints,
	int * d_emptytriangles,
	int emptypointsLength,
	int emptytrianglesLength,
	int numberofemptypoints,
	int numberofemptytriangles,
	int numberofsubseg,
	int numberofactive,
	REAL theta,
	int iteration
);

__global__ void kernelUpdateSteinerNeighbors
(
	REAL2 * d_pointlist,
	int	* d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_internalmarker,
	int * d_internallist,
	int * d_sinks,
	int numberofactive,
	int run_mode,
	REAL theta,
	int debug_iter
);

__global__ void kernelUpdateSteinerOnsegs
(
	int * d_encmarker,
	int * d_internallist,
	int numberofactive
);

__global__ void kernelUpdatePStatus2Old(
	PStatus * d_PStatus,
	int last_point
);

// split bad elements
__global__ void kernelComputeSplittingPointAndPriority(
	REAL2 * d_pointlist,
	int * d_subseg2tri,
	int * d_trianglelist,
	int * d_neighborlist,
	int	* d_tri2subseg,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs,
	REAL2 * d_insertpt,
#ifdef GQM2D_PRIORITY_SIZE
	REAL * d_priorityreal,
#endif
	REAL offConstant,
	bool offCenter
);

__global__ void kernelModifyPriority(
	REAL* d_priorityreal,
	int* d_priority,
	REAL offset0,
	REAL offset1,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs
);

__global__ void kernelLocateSplittingPoints(
	REAL2 * d_pointlist,
	int * d_subseg2tri,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_insertpt,
#ifdef GQM2D_PRIORITY_SIZE
	uint64 * d_trimarker,
	int * d_priority,
#else
	int * d_trimarker,
#endif
	int * d_sinks,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs
);

__global__ void kernelFastCavityCheck(
	REAL2 * d_pointlist,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_insertpt,
#ifdef GQM2D_PRIORITY_SIZE
	uint64 * d_trimarker,
	int * d_priority,
#else
	int * d_trimarker,
#endif
	int * d_sinks,
	int * d_elementlist,
	int numberofelements
);

__global__ void kernelMarkOnSegs
(
	int *d_trianglelist,
	int *d_neighborlist,
	int * d_tri2subseg,
	int* d_segmarker,
	int	*d_sinks,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs
);

__global__ void kernelResetNeighborMarker(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_sinks,
	int * d_threadlist,
	int numberofsteiners
);

__global__ void kernelInsertSplittingPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
#ifdef GQM2D_PRIORITY_SIZE
	REAL* d_pointpriority,
	REAL* d_priorityreal,
#endif
	int * d_subseg2tri,
	int * d_subseg2seg,
	int * d_encmarker,
	int * d_segmarker,
	int	* d_trianglelist,
	int * d_neighborlist,
	int	* d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_insertpt,
	int	* d_sinks,
	int * d_emptypoints,
	int * d_emptytriangles,
	int * d_elementlist,
	int * d_threadlist,
	int emptypointsLength,
	int emptytrianglesLength,
	int numberofemptypoints,
	int numberofemptytriangles,
	int numberofsubseg,
	int numberofwonsegs,
	int numberofsteiners,
	int encmode,
	REAL theta
);

__global__ void kernelUpdateNeighbors
(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_segmarker,
	int	* d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_sinks,
	int * d_emptypoints,
	int * d_threadlist,
	int emptypointsLength,
	int numberofemptypoints,
	int numberofwonsegs,
	int numberofsteiners,
	int encmode,
	REAL theta
);

__global__ void markBadSubsegsAsEncroached
(
	int * d_encmarker,
	int * d_segmarker,
	int numberofsubsegs
);

// main refinement routine
void updateNeighborsFormat2Otri
(
	IntD	&t_trianglelist,
	IntD	&t_neighborlist,
	int numberoftriangles
);

void initSubsegs
(
	IntD	&t_segmentlist,
	IntD	&t_subseg2tri,
	IntD    &t_trianglelist,
	IntD    &t_neighborlist,
	IntD    &t_tri2subseg,
	int numberofsegments
);

void markAllEncsegs(
	Real2D	&t_pointlist,
	IntD	&t_trianglelist,
	IntD	&t_neighborlist,
	IntD	&t_subseg2tri,
	IntD	&t_encmarker,
	int	numberofsubseg,
	int run_mode,
	REAL theta
);

void updatePStatus2Old
(
	PStatusD &t_PStatus,
	int last_point
);

int compactTrianlgeList
(
	IntD		&t_trianglelist,
	IntD       &t_neighborlist,
	TStatusD	&t_TStatus,
	IntD		&t_subseg2tri,
	IntD       &t_valid,
	IntD       &t_prefix,
	int		numberoftriangles,
	int		numberofsubsegs,
	int		maxnumofblock
);

int compactPointList
(
	IntD		&t_trianglelist,
	Real2D		&t_pointlist,
	PStatusD	&t_PStatus,
	IntD       &t_valid,
	IntD       &t_prefix,
	int		numberoftriangles,
	int		numberofpoints,
	int		maxnumofblock
);

void compactSegmentList
(
	IntD &t_trianglelist,
	IntD &t_subseg2tri,
	IntD &t_segmentlist,
	int numberofsubsegs,
	int maxnumofblock
);

void updateNeighborsFormat2Int
(
	int *d_neighborlist,
	int numberoftriangles,
	int maxnumofblock
);