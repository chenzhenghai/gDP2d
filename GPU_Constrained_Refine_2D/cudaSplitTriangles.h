#pragma once

#include "cudaThrust.h"

void splitTriangles(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	IntD	 &t_tri2subseg,
	TStatusD &t_TStatus,
	IntD	 &t_subseg2tri,
	IntD	 &t_encmarker,
	IntD	 &t_enclist,
	IntD	 &t_internalmarker,
	IntD	 &t_internallist,
	Real2D	 &t_TCenter,
	IntD	 &t_sinks,
	IntD	 &t_Priority,
	IntD	 &t_trimarker,
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
	REAL theta,
	int debug_iter
);