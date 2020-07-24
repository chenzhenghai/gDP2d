#include <time.h>
#include "cudaRefine.h"
#include "cudaFlipFlop.h"
#include "cudaSplitEncsegs.h"
#include "cudaSplitTriangles.h"
#include "cudaSplitElements.h"
#include "cudaMesh.h"

void GPU_Refine_Quality(triangulateio *input, triangulateio *result, double input_theta, double input_size, InsertPolicy insertpolicy, DeletePolicy deletepolicy, int encmode, int runmode,
	int filtermode, int unifymode, int debug_iter, PStatus **ps_debug, TStatus **ts_debug)
{
	if(encmode == 0)
		printf("Encroachment mode: Chew\n");
	else if(encmode == 1)
		printf("Encroachment mode: Ruppert\n");
	else
	{
		printf("Unknown encroachment mode: %d\n", encmode);
		exit(0);
	}

	if(runmode == 0)
		printf("Running mode: without vectorization\n");
	else if(runmode == 1)
		printf("Running mode: with vectorization\n");
	else
	{
		printf("Unknown runnning mode: %d\n", runmode);
		exit(0);
	}

	/************************************/
	/* 0. Initialization				*/
	/************************************/

	/* Set up timer */
	StopWatchInterface *inner_timer = 0;
	sdkCreateTimer(&inner_timer);

	///* Set up double precise */
	//_control87(_PC_53, _MCW_PC); /* Set FPU control word for double precision. */

	/* Initialize memory */

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// Input variables and arrays
	int numberofpoints = input->numberofpoints;
	int numberoftriangles = input->numberoftriangles;
	int numberofsegments = input->numberofsegments;

	Real2D t_pointlist((REAL2 *)input->pointlist, (REAL2 *)input->pointlist + numberofpoints);
	IntD t_trianglelist(input->trianglelist, input->trianglelist + numberoftriangles * 3);
	IntD t_neighborlist(input->neighborlist, input->neighborlist + numberoftriangles * 3);
	IntD t_segmentlist(input->segmentlist, input->segmentlist + numberofsegments * 2);

	cudaexactinit();

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM2D_PROFILING
	clock_t tv[2];
	cudaDeviceSynchronize();
	tv[0] = clock();
#endif

	// Transfer neighbor format to orientation triangles (otri)
	updateNeighborsFormat2Otri(
		t_trianglelist,
		t_neighborlist,
		numberoftriangles);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM2D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf(" updateNeighborsFormat2Otri time = %f\n", (REAL)(tv[1] - tv[0]));
#endif

	// Memory variables
	int pointblock = numberofpoints*0.5;	/* Number of points	allocated at once. */
	int triblock = 2 * pointblock;			/* Number of triangles allocated at once. */

											// Iteration variables
	int last_point = numberofpoints;
	int last_triangle = numberoftriangles;
	int last_subseg = numberofsegments;

	int numberofbad;
	int numberofemptypoints;
	int numberofemptytriangles;

	bool offcenter;
	double offconstant;
	if (insertpolicy == Circumcenter)
	{
		offcenter = false;
		offconstant = 0.0;
	}
	else if (insertpolicy == Offcenter)
	{
		offcenter = true;
		double cos_good = cos(input_theta*PI / 180);
		offconstant = 0.475 * sqrt((1.0 + cos_good) / (1.0 - cos_good));
	}

	// Pre-allocate slots for insertion
	int presize = last_point + pointblock;
	int tresize = last_triangle + triblock;

	t_pointlist.resize(presize);
	t_trianglelist.resize(3 * tresize);
	t_neighborlist.resize(3 * tresize);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// Fixed arrays (values maintained all the time)
	PStatusD t_PStatus(presize);
	TStatusD t_TStatus(tresize);

	IntD t_subseg2tri(input->segment2trilist, input->segment2trilist + last_subseg);
	IntD t_subseg2seg(last_subseg);
	IntD t_encmarker(last_subseg, -1); // initialize to non-encroached
	IntD t_tri2subseg(3 * tresize, -1);

	IntD t_emptypoints(numberofpoints);
	IntD t_emptytriangles(numberoftriangles);

	IntD errorseg, errortri; // error tags for segments and triangles

	// Re-usuable array
	IntD t_list0(numberoftriangles);
	IntD t_list1(numberoftriangles);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// Init PStatus
	PStatus inputPoint;
	PStatus emptyPoint;
	emptyPoint.setDeleted();

	thrust::fill(t_PStatus.begin(), t_PStatus.begin() + numberofpoints, inputPoint);
	thrust::fill(t_PStatus.begin() + numberofpoints, t_PStatus.end(), emptyPoint);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// Init TStatus
	TStatus inputTri(false, false, true);		/* not null, need to check */
	TStatus emptyTri(true, false, false);		/* empty triangles */

	thrust::fill(t_TStatus.begin(), t_TStatus.begin() + numberoftriangles, inputTri);
	thrust::fill(t_TStatus.begin() + numberoftriangles, t_TStatus.end(), emptyTri);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM2D_PROFILING
	cudaDeviceSynchronize();
	tv[0] = clock();
#endif

	// Init subseg2tri, subseg2seg and tri2subseg
	initSubsegs(
		t_segmentlist,
		t_subseg2tri,
		t_trianglelist,
		t_neighborlist,
		t_tri2subseg,
		last_subseg);

	thrust::sequence(t_subseg2seg.begin(), t_subseg2seg.begin() + last_subseg, 0); // Init subsegment to contain itself

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM2D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf(" initSubsegs time = %f\n", (REAL)(tv[1] - tv[0]));
#endif

	// Init empty lists
	numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
	numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// Get timer.
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);
	printf("0. Initialization time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	//exit(0);

	/************************************/
	/* 1. Processing					*/
	/************************************/

	// mark all encroached subsegments
	markAllEncsegs(
		t_pointlist,
		t_trianglelist,
		t_neighborlist,
		t_subseg2tri,
		t_encmarker,
		last_subseg,
		encmode,
		input_theta);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM2D_DEBUG_3
	// check if encroachment markers are updated correctly
	{
		printf("After Initialization: Error Checking....\n");
		int * debug_em = new int[last_subseg];
		int * debug_tl = new int[3 * last_triangle];
		int * debug_nl = new int[3 * last_triangle];
		REAL2 * debug_pl = new REAL2[last_point];
		int * debug_st = new int[last_subseg];
		cudaMemcpy(debug_em, thrust::raw_pointer_cast(&t_encmarker[0]), sizeof(int)*last_subseg, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * last_triangle, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_nl, thrust::raw_pointer_cast(&t_neighborlist[0]), sizeof(int) * 3 * last_triangle, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)*last_point, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_st, thrust::raw_pointer_cast(&t_subseg2tri[0]), sizeof(int)*last_subseg, cudaMemcpyDeviceToHost);

		for (int i = 0; i< last_subseg; i++)
		{
			int otri = debug_st[i];
			int tri = otri >> 2;
			int ori = otri & 3;
			int p[3];
			REAL2 v[3];
			p[0] = debug_tl[3 * tri + (ori + 1) % 3];
			p[1] = debug_tl[3 * tri + (ori + 2) % 3];
			p[2] = debug_tl[3 * tri + ori];
			v[0] = debug_pl[p[0]];
			v[1] = debug_pl[p[1]];
			v[2] = debug_pl[p[2]];
			bool tag = false; // indicate if this segment is encroached or not

			REAL goodcoss = cos(theta * PI / 180.0);
			goodcoss *= goodcoss;
			REAL dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
				(v[0].y - v[2].y)*(v[1].y - v[2].y);

			if (dotproduct < 0.0) // angle > 90
			{
				// here, we use diametral lens to speedup the algorithm
				if (encmode || dotproduct * dotproduct >=
					(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
					((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
					((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
					tag = true;
			}

			otri = debug_nl[3 * tri + ori];
			if (otri != -1)
			{
				tri = otri >> 2;
				ori = otri & 3;
				p[2] = debug_tl[3 * tri + ori];
				v[2] = debug_pl[p[2]];
				dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
					(v[0].y - v[2].y)*(v[1].y - v[2].y);
				if (dotproduct < 0.0) // angle > 90
				{
					// here, we use diametral lens to speedup the algorithm
					if (encmode || dotproduct * dotproduct >=
						(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
						((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
						((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
						tag = true;
				}
			}

			if (debug_em[i] == -1 && tag)
				printf("Segment %d: I am encroached but marked as non-encroached\n", i);

			if (debug_em[i] == 0 && !tag)
				printf("Segment %d: I am not encroached but marked as encroached\n", i);

		}
		delete[] debug_em;
		delete[] debug_tl;
		delete[] debug_nl;
		delete[] debug_pl;
		delete[] debug_st;
		printf("Finished Checking\n");
	}
#endif

	if (runmode == 0)
	{
		// Flexible arrays 
		IntD t_trimarker(tresize);
		Real2D t_TCenter(numberoftriangles);

		IntD t_list2(numberoftriangles);
		IntD t_list3(numberoftriangles);
		IntD t_list4(numberoftriangles);
		IntD t_list5(numberoftriangles);

		/* Split all encroached subsegments */

#ifdef GQM2D_LOOP_PROFILING
		double time_segs = 0;
		double time_tri = 0;
#endif

		// Reset and start timer.
		cudaDeviceSynchronize();
		sdkResetTimer(&inner_timer);
		sdkStartTimer(&inner_timer);

		// split all encroached subsegments until no more subsegments are encroached
		printf("Splitting encroached subsegments....\n");
		splitEncsegs(
			t_pointlist,
			t_PStatus,
			t_trianglelist,
			t_neighborlist,
			t_tri2subseg,
			t_TStatus,
			t_segmentlist,
			t_subseg2tri,
			t_subseg2seg,
			t_encmarker, // encroached marker
			t_list0, // encroached list
			t_list1, // internal marker
			t_list2, // internal list
			t_trimarker,// flipBy
			t_list3, // flipActive,
			t_list4, // linklist
			t_list5, // linkslot
			t_emptypoints,
			t_emptytriangles,
			pointblock,
			triblock,
			&numberofemptypoints,
			&numberofemptytriangles,
			&last_point,
			&last_triangle,
			&last_subseg,
			encmode,
			input_theta,
			-1);

		// Check if triangles have duplicate vertices
#ifdef GQM2D_DEBUG_3
		{
			printf("After Splitting initial encroached segments: Error Checking....\n");
			int * debug_tl = new int[3 * last_triangle];
			REAL2 * debug_pl = new REAL2[last_point];
			TStatus * debug_ts = new TStatus[last_triangle];
			cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * last_triangle, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)*last_point, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_ts, thrust::raw_pointer_cast(&t_TStatus[0]), sizeof(TStatus)*last_triangle, cudaMemcpyDeviceToHost);
			for (int i = 0; i < last_triangle; i++)
			{
				if (!debug_ts[i].isNull())
				{
					bool errorflag = false;
					int p[3];
					REAL2 v[3];
					for (int j = 0; j < 3; j++)
					{
						p[j] = debug_tl[3 * i + j];
						v[j] = debug_pl[p[j]];
					}
					for (int j = 0; j < 2; j++)
					{
						for (int k = j + 1; k < 3; k++)
						{
							if (v[j].x == v[k].x && v[j].y == v[k].y)
							{
								errorflag = true;
							}
						}
					}
					if (errorflag)
						printf("Tri %d: Duplicate vertice\n", i);
				}
			}
			delete[] debug_tl;
			delete[] debug_pl;
			delete[] debug_ts;
			printf("Finished Checking\n");
		}
#endif

		cudaDeviceSynchronize();
		sdkStopTimer(&inner_timer);
		printf("1.1 Split initial encroached segments time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
#ifdef GQM2D_LOOP_PROFILING
		time_segs += sdkGetTimerValue(&inner_timer);
#endif

		/* Enforce triangle quality */
		printf("Splitting bad triangles....\n");
		int iteration = 0;
		int iter_numofpt = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
		printf("  iteration -1: number of points = %d, 0\n", iter_numofpt);
		StopWatchInterface *iter_timer = 0;
		sdkCreateTimer(&iter_timer);
		sdkResetTimer(&iter_timer);
		sdkStartTimer(&iter_timer);
		while (true)
		{
			// Update bad triangles list
			numberofbad = updateActiveListToBadTriangles(
				t_pointlist,
				t_PStatus,
				t_trianglelist,
				t_neighborlist,
				t_segmentlist,
				t_subseg2seg,
				t_tri2subseg,
				t_TStatus,
				t_list0,
				t_list1, // output: bad triangles list
				last_triangle,
				input_theta,
				input_size);

			if (numberofbad == 0)
				break;
			//printf("numberofbad = %d\n",numberofbad);

#ifdef GQM2D_LOOP_PROFILING
			cudaDeviceSynchronize();
			sdkResetTimer(&inner_timer);
			sdkStartTimer(&inner_timer);
#endif

			// Split all bad triangles
			splitTriangles(
				t_pointlist,
				t_PStatus,
				t_trianglelist,
				t_neighborlist,
				t_tri2subseg,
				t_TStatus,
				t_subseg2tri,
				t_encmarker,
				t_list0, // encroached list
				t_list1, // internal marker(initially bad triangles list)
				t_list2, // internal list
				t_TCenter, // steiner points
				t_list3,   // sinks
				t_list4, // shortest
				t_trimarker,
				t_emptypoints,
				t_emptytriangles,
				pointblock,
				triblock,
				&numberofemptypoints,
				&numberofemptytriangles,
				&last_point,
				&last_triangle,
				&last_subseg,
				offconstant,
				offcenter,
				encmode,
				input_theta,
				iteration);

#ifdef GQM2D_LOOP_PROFILING
			cudaDeviceSynchronize();
			sdkStopTimer(&inner_timer);
			printf("   split triangles time = %.3f\n", sdkGetTimerValue(&inner_timer));
			time_tri += sdkGetTimerValue(&inner_timer);
#endif

#ifdef GQM2D_DEBUG
			if (iteration == debug_iter)
				break;
#endif

#ifdef GQM2D_DEBUG_3
			{
				int * debug_tl = new int[3 * last_triangle];
				REAL2 * debug_pl = new REAL2[last_point];
				TStatus * debug_ts = new TStatus[last_triangle];
				cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * last_triangle, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)*last_point, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_ts, thrust::raw_pointer_cast(&t_TStatus[0]), sizeof(TStatus)*last_triangle, cudaMemcpyDeviceToHost);
				for (int i = 0; i < last_triangle; i++)
				{
					if (!debug_ts[i].isNull())
					{
						bool errorflag = false;
						int p[3];
						REAL2 v[3];
						for (int j = 0; j < 3; j++)
						{
							p[j] = debug_tl[3 * i + j];
							v[j] = debug_pl[p[j]];
						}
						for (int j = 0; j < 2; j++)
						{
							for (int k = j + 1; k < 3; k++)
							{
								if (v[j].x == v[k].x && v[j].y == v[k].y)
								{
									errorflag = true;
								}
							}
						}
						if (errorflag)
							printf(" After splitTriangles - Tri %d: Duplicate vertice\n", i);
					}
				}
				delete[] debug_tl;
				delete[] debug_pl;
				delete[] debug_ts;
			}
#endif

#ifdef GQM2D_LOOP_PROFILING
			sdkResetTimer(&inner_timer);
			sdkStartTimer(&inner_timer);
#endif

			// Split all encroached subsegs
			splitEncsegs(
				t_pointlist,
				t_PStatus,
				t_trianglelist,
				t_neighborlist,
				t_tri2subseg,
				t_TStatus,
				t_segmentlist,
				t_subseg2tri,
				t_subseg2seg,
				t_encmarker, // encroached marker
				t_list0, // encroached list
				t_list1, // internal marker
				t_list2, // internal list
				t_trimarker,// flipBy
				t_list3, // flipActive,
				t_list4, // linklist
				t_list5, // linkslot
				t_emptypoints,
				t_emptytriangles,
				pointblock,
				triblock,
				&numberofemptypoints,
				&numberofemptytriangles,
				&last_point,
				&last_triangle,
				&last_subseg,
				encmode,
				input_theta,
				iteration);

#ifdef GQM2D_LOOP_PROFILING
			cudaDeviceSynchronize();
			sdkStopTimer(&inner_timer);
			printf("   split subsegments time = %.3f\n", sdkGetTimerValue(&inner_timer));
			time_segs += sdkGetTimerValue(&inner_timer);
#endif

			updatePStatus2Old(
				t_PStatus,
				last_point);

#ifdef GQM2D_DEBUG_3
			{
				int * debug_tl = new int[3 * last_triangle];
				REAL2 * debug_pl = new REAL2[last_point];
				TStatus * debug_ts = new TStatus[last_triangle];
				cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * last_triangle, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)*last_point, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_ts, thrust::raw_pointer_cast(&t_TStatus[0]), sizeof(TStatus)*last_triangle, cudaMemcpyDeviceToHost);
				for (int i = 0; i < last_triangle; i++)
				{
					if (!debug_ts[i].isNull())
					{
						bool errorflag = false;
						int p[3];
						REAL2 v[3];
						for (int j = 0; j < 3; j++)
						{
							p[j] = debug_tl[3 * i + j];
							v[j] = debug_pl[p[j]];
						}
						for (int j = 0; j < 2; j++)
						{
							for (int k = j + 1; k < 3; k++)
							{
								if (v[j].x == v[k].x && v[j].y == v[k].y)
								{
									errorflag = true;
								}
							}
						}
						if (errorflag)
							printf(" After splitEncsegs - Tri %d: Duplicate vertice\n", i);
					}
				}
				delete[] debug_tl;
				delete[] debug_pl;
				delete[] debug_ts;
			}
#endif

#ifdef GQM2D_LOOP_PROFILING
			iter_numofpt = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
			printf("  iteration = %d, number of point = %d, %.3f\n", iteration, iter_numofpt, sdkGetTimerValue(&iter_timer));
#else
			printf("  Iteration = %d, numberofbad = %d\n", iteration, numberofbad);
#endif

#ifdef GQM2D_DEBUG
			if (iteration == debug_iter)
				break;
#endif

			iteration++;
		}

		// Get timer.
		cudaDeviceSynchronize();
		sdkStopTimer(&iter_timer);
#ifdef GQM2D_LOOP_PROFILING
		printf(" Time for SplitEncsegs = %.3f ms\n", time_segs);
		printf(" Time for SplitTriangles = %.3f ms\n", time_tri);
		printf(" Ratio = %.3f\n", time_segs / time_tri);
#endif
		printf("1.2 Split bad triangles time = %.3f ms\n", sdkGetTimerValue(&iter_timer));

		// deallocate memeory
		//t_trimarker.clear();
		//t_TCenter.clear();
		//t_list2.clear();
		//t_list3.clear();
		//t_list4.clear();
		//t_list5.clear();
		//t_trimarker.shrink_to_fit();
		//t_TCenter.shrink_to_fit();
		//t_list2.shrink_to_fit();
		//t_list3.shrink_to_fit();
		//t_list4.shrink_to_fit();
		//t_list5.shrink_to_fit();
	}
	else
	{
		// Flexible arrays 
		IntD t_trimarker(tresize);
		IntD t_list2(numberoftriangles);
		IntD t_list3(numberoftriangles);
		IntD t_list4(numberoftriangles);
		IntD t_list5(numberoftriangles);

		// split all encroached subsegments until no more subsegments are encroached
		printf("Splitting encroached subsegments....\n");
		splitEncsegs(
			t_pointlist,
			t_PStatus,
			t_trianglelist,
			t_neighborlist,
			t_tri2subseg,
			t_TStatus,
			t_segmentlist,
			t_subseg2tri,
			t_subseg2seg,
			t_encmarker, // encroached marker
			t_list0, // encroached list
			t_list1, // internal marker
			t_list2, // internal list
			t_trimarker,// flipBy
			t_list3, // flipActive,
			t_list4, // linklist
			t_list5, // linkslot
			t_emptypoints,
			t_emptytriangles,
			pointblock,
			triblock,
			&numberofemptypoints,
			&numberofemptytriangles,
			&last_point,
			&last_triangle,
			&last_subseg,
			encmode,
			input_theta,
			-1);

		t_trimarker.resize(0);
		t_list2.resize(0);
		t_list3.resize(0);
		t_list4.resize(0);
		t_list5.resize(0);
		t_trimarker.shrink_to_fit();
		t_list2.shrink_to_fit();
		t_list3.shrink_to_fit();
		t_list4.shrink_to_fit();
		t_list5.shrink_to_fit();

		// Split all bad elements
		splitElements(
			t_pointlist,
			t_PStatus,
			t_segmentlist,
			t_subseg2tri,
			t_subseg2seg,
			t_encmarker,
			t_trianglelist,
			t_neighborlist,
			t_tri2subseg,
			t_TStatus,
			t_emptypoints,
			t_emptytriangles,
			pointblock,
			triblock,
			&numberofemptypoints,
			&numberofemptytriangles,
			&last_point,
			&last_triangle,
			&last_subseg,
			offconstant,
			offcenter,
			encmode,
			filtermode,
			unifymode,
			input_theta,
			input_size);
	}

	/************************************/
	/* 2. Getting result				*/
	/************************************/

	/* Copy results to host */

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// deallocate memeory
	//t_encmarker.clear();
	//t_encmarker.shrink_to_fit();

	// triangles
	last_triangle =
		compactTrianlgeList(
			t_trianglelist,
			t_neighborlist,
			t_TStatus,
			t_subseg2tri,
			t_list0,
			t_list1,
			last_triangle,
			last_subseg,
			5000);

	last_point =
		compactPointList(
			t_trianglelist,
			t_pointlist,
			t_PStatus,
			t_list0,
			t_list1,
			last_triangle,
			last_point,
			5000);

	compactSegmentList(
		t_trianglelist,
		t_subseg2tri,
		t_segmentlist,
		last_subseg,
		5000);

	// points
	result->numberofpoints = last_point;
	result->pointlist = new double[2 * last_point];
	cudaMemcpy(result->pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 2 * last_point * sizeof(double), cudaMemcpyDeviceToHost);

	// triangles
	result->numberoftriangles = last_triangle;
	result->trianglelist = new int[3 * last_triangle];
	cudaMemcpy(result->trianglelist, thrust::raw_pointer_cast(&t_trianglelist[0]), 3 * last_triangle * sizeof(int), cudaMemcpyDeviceToHost);

	// neighbors
	updateNeighborsFormat2Int(
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		last_triangle,
		5000);

	result->neighborlist = new int[3 * last_triangle];
	cudaMemcpy(result->neighborlist, thrust::raw_pointer_cast(&t_neighborlist[0]), 3 * last_triangle * sizeof(int), cudaMemcpyDeviceToHost);

	// segments
	result->numberofsegments = last_subseg;
	result->segmentlist = new int[2 * last_subseg];
	cudaMemcpy(result->segmentlist, thrust::raw_pointer_cast(&t_segmentlist[0]), 2 * last_subseg * sizeof(int), cudaMemcpyDeviceToHost);

	// debug
	if (ps_debug != NULL)
	{
		*ps_debug = new PStatus[last_point];
		cudaMemcpy(*ps_debug, thrust::raw_pointer_cast(&t_PStatus[0]), last_point * sizeof(PStatus), cudaMemcpyDeviceToHost);
	}
	if (ts_debug != NULL)
	{
		*ts_debug = new TStatus[last_triangle];
		cudaMemcpy(*ts_debug, thrust::raw_pointer_cast(&t_TStatus[0]), last_triangle * sizeof(TStatus), cudaMemcpyDeviceToHost);
	}

	//t_pointlist.clear();
	//t_segmentlist.clear();
	//t_trianglelist.clear();
	//t_neighborlist.clear();
	//t_pointlist.shrink_to_fit();
	//t_segmentlist.shrink_to_fit();
	//t_trianglelist.shrink_to_fit();
	//t_neighborlist.shrink_to_fit();

	// Get timer.
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);
	printf("2. Get result time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
}