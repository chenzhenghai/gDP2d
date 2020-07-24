#pragma once
#include <time.h>
#include "cudaThrust.h"

#define GQM2D_QUIET
#define GQM2D_ITER_PROFILING

void splitElements(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_segmentlist,
	IntD	 &t_subseg2tri,
	IntD	 &t_subseg2seg,
	IntD	 &t_encmarker,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	IntD	 &t_tri2subseg,
	TStatusD &t_TStatus,
	IntD	 &t_emptypoints,
	IntD	 &t_emptytriangles,
	int pointblock,
	int triblock,
	int * numberofemptypoints,
	int * numberofemptytriangles,
	int * numberofpoints,
	int	* numberoftriangles,
	int	* numberofsubseg,
	int offconstant,
	int offcenter,
	int encmode,
	int filtermode,
	int unifymode,
	REAL theta,
	REAL size
);