#pragma once

#include "cudaMesh.h"
#include "cudaThrust.h"

void flipFlop
(
	Real2D			&t_pointlist,
	PStatusD		&t_PStatus,
	IntD			&t_trianglelist,
	IntD           &t_neighborlist,
	IntD			&t_tri2subseg,
	TStatusD		&t_TStatus,
	IntD			&t_subseg2tri,
	IntD           &t_flipBy,
	IntD           &t_active,
	IntD			&t_encmarker,
	IntD			&t_linklist,
	IntD			&t_linkslot,
	int			numberoftriangles,
	int encmode,
	REAL theta,
	int iteration0,
	int iteration1
);

void flipFlop
(
	Real2D			&t_pointlist,
	PStatusD		&t_PStatus,
#ifdef GQM2D_PRIORITY_SIZE
	RealD			&t_pointpriority,
#endif
	IntD			&t_trianglelist,
	IntD           &t_neighborlist,
	IntD			&t_tri2subseg,
	TStatusD		&t_TStatus,
	IntD			&t_subseg2tri,
	IntD           &t_flipBy,
	IntD           &t_active,
	IntD			&t_encmarker,
	IntD            &t_segmarker,
	IntD			&t_linklist,
	IntD			&t_linkslot,
	int			numberoftriangles,
	int encmode,
	REAL theta,
	int iteration0,
	int iteration1
);