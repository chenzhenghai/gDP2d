#pragma once

#include "cudaThrust.h"

void splitEncsegs(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	IntD	 &t_tri2subseg,
	TStatusD &t_TStatus,
	IntD	 &t_segmentlist,
	IntD	 &t_subseg2tri,
	IntD	 &t_subseg2seg,
	IntD	 &t_encmarker,
	IntD	 &t_enclist,
	IntD	 &t_internalmarker,
	IntD	 &t_internallist,
	IntD	 &t_flipBy,
	IntD	 &t_flipActive,
	IntD	 &t_linklist,
	IntD	 &t_linkslot,
	IntD	 &t_emptypoints,
	IntD	 &t_emptytriangles,
	int pointblock,
	int triblock,
	int * numberofemptypoints,
	int * numberofemptytriangles,
	int * numberofpoints,
	int	* numberoftriangles,
	int	* numberofsubseg,
	int encmode,
	REAL theta,
	int debug_iter
);